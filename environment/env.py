import random
import time
import os
import warnings

import gym
from gym import spaces
from gym.utils import seeding
from PIL import Image
import numpy as np

from config import INPUT_DIM, MIN_STEERING, MAX_STEERING, JERK_REWARD_WEIGHT, MAX_STEERING_DIFF
from config import ROI, THROTTLE_REWARD_WEIGHT, MAX_THROTTLE, MIN_THROTTLE, REWARD_CRASH, CRASH_SPEED_WEIGHT
from environment.carla.client import make_carla_client, CarlaClient 
from environment.carla.tcp import TCPConnectionError
from environment.carla.settings import CarlaSettings
from environment.carla.sensor import Camera
from environment.carla.carla_server_pb2 import Control

class Env(gym.Env):
    def __init__(self, client, vae=None, min_throttle=0.2, max_throttle=0.5, n_command_history=0, frame_skip=2, n_stack=1, action_lambda=0.5):
        self.client = client
        # save last n commands
        self.n_commands = 2
        self.n_command_history = n_command_history
        self.command_history = np.zeros((1, self.n_commands * n_command_history))
        self.n_stack = n_stack
        self.stacked_obs = None

        # assumes that we are using VAE input
        self.vae = vae
        self.z_size = None
        if vae is not None:
            self.z_size = vae.z_size
        
        self.observation_space = spaces.Box(low=np.finfo(np.float32).min,
            high=np.finfo(np.float32).max,
            shape=(1, self.z_size + self.n_commands * n_command_history),
            dtype=np.float32)
        self.action_space = spaces.Box(low=np.array([-MAX_STEERING, -1]),
            high=np.array([MAX_STEERING, 1]),
            dtype=np.float32)

        self.min_throttle = min_throttle
        self.max_throttle = max_throttle
        self.frame_skip = frame_skip
        self.action_lambda = action_lambda
        self.last_throttle = 0.0
        self.seed()

    def jerk_penalty(self):
        """
        Add a continuity penalty to limit jerk.
        :return: (float)
        """
        jerk_penalty = 0
        if self.n_command_history > 1:
            # Take only last command into account
            for i in range(1):
                steering = self.command_history[0, -2 * (i + 1)]
                prev_steering = self.command_history[0, -2 * (i + 2)]
                steering_diff = (prev_steering - steering) / (MAX_STEERING - MIN_STEERING)

                if abs(steering_diff) > MAX_STEERING_DIFF:
                    error = abs(steering_diff) - MAX_STEERING_DIFF
                    jerk_penalty += JERK_REWARD_WEIGHT * (error ** 2)
                else:
                    jerk_penalty += 0
        return jerk_penalty

    def postprocessing_step(self, action, observation, reward, done, info):
        """
        Update the reward (add jerk_penalty if needed), the command history
        and stack new observation (when using frame-stacking).

        :param action: ([float])
        :param observation: (np.ndarray)
        :param reward: (float)
        :param done: (bool)
        :param info: (dict)
        :return: (np.ndarray, float, bool, dict)
        """
        # Update command history
        if self.n_command_history > 0:
            self.command_history = np.roll(self.command_history, shift=-self.n_commands, axis=-1)
            self.command_history[..., -self.n_commands:] = action
            observation = np.concatenate((observation, self.command_history), axis=-1)

        jerk_penalty = self.jerk_penalty()
        # Cancel reward if the continuity constrain is violated
        if jerk_penalty > 0 and reward > 0:
            reward = 0
        reward -= jerk_penalty

        if self.n_stack > 1:
            self.stacked_obs = np.roll(self.stacked_obs, shift=-observation.shape[-1], axis=-1)
            if done:
                self.stacked_obs[...] = 0
            self.stacked_obs[..., -observation.shape[-1]:] = observation
            return self.stacked_obs, reward, done, info

        return observation, reward, done, info

    # def step(self, action):
    #     while True:
    #         try:
    #             with make_carla_client('localhost', 2000) as client:
    #                 self.client = client
    #                 return self.step_env(action)
    #         except TCPConnectionError as error:
    #             print(error)
    #             time.sleep(1.0)

    def step(self, action):
        # Convert from [-1, 1] to [0, 1]
        t = (action[1] + 1) / 2
        # Convert from [0, 1] to [min, max]
        action[1] = (1 - t) * self.min_throttle + self.max_throttle * t

        # Clip steering angle rate to enforce continuity
        if self.n_command_history > 0:
            prev_steering = self.command_history[0, -2]
            max_diff = (MAX_STEERING_DIFF - 1e-5) * (MAX_STEERING - MIN_STEERING)
            diff = np.clip(action[0] - prev_steering, -max_diff, max_diff)
            action[0] = prev_steering + diff

        control = Control()
        control.throttle = action[1]
        control.steer = action[0]
        control.brake = 0
        control.hand_brake = 0
        control.reverse = 0

        # Repeat action if using frame_skip
        for _ in range(self.frame_skip):
            self.client.send_control(control)
            measurements, sensor_data = self.client.read_data()
            im = sensor_data['CameraRGB'].data
            im = np.array(im)
            im = im[:, :, ::-1] # convert to BGR
            observation = self.vae.encode_from_raw_image(im)
            reward, done = self.reward(measurements, action)

        self.last_throttle = action[1]

        return self.postprocessing_step(action, observation, reward, done, {})

    def reset(self):
        print("Start to reset env")
        settings = CarlaSettings()
        settings.set(
            SynchronousMode=True,
            SendNonPlayerAgentsInfo=False,
            NumberOfVehicles=0,
            NumberOfPedestrians=0,
            # WeatherId=random.choice([1]),
            WeatherId=random.choice([1]),
            QualityLevel='Epic'
        )
        settings.randomize_seeds()
        camera = Camera('CameraRGB')
        camera.set(FOV=100)
        camera.set_image_size(160, 120)
        camera.set_position(2.0, 0.0, 1.4)
        camera.set_rotation(-15.0, 0, 0)
        settings.add_sensor(camera)
        observation = None

        # self.start_client()
        scene = self.client.load_settings(settings)
        number_of_player_starts = len(scene.player_start_spots)
        player_start = random.randint(0, max(0, number_of_player_starts - 1))
        self.client.start_episode(player_start)

        # action = np.zeros(2)
        # observation, _, _,_ = self.step(action)

        measurements, sensor_data = self.client.read_data()
        im = sensor_data['CameraRGB'].data
        im = np.array(im)
        im = im[:, :, ::-1] # convert to BGR
        observation = self.vae.encode_from_raw_image(im)

        self.command_history = np.zeros((1, self.n_commands * self.n_command_history))

        if self.n_command_history > 0:
            observation = np.concatenate((observation, self.command_history), axis=-1)

        if self.n_stack > 1:
            self.stacked_obs[...] = 0
            self.stacked_obs[..., -observation.shape[-1]:] = observation
            return self.stacked_obs

        print('reset finished')
        return observation


    def reward(self, measurements, action):
        """
        :param measurements:
        :return: reward, done
        """
        done = False

        """distance"""

        """speed"""
        # # In the wayve.ai paper, speed has been used as reward
        # SPEED_REWARD_WEIGHT = 0.1
        # speed_reward = SPEED_REWARD_WEIGHT*measurements.player_measurements.forward_speed

        """road"""
        if measurements.player_measurements.intersection_offroad > 0.2 or measurements.player_measurements.intersection_otherlane > 0.2:
            norm_throttle = (self.last_throttle - MIN_THROTTLE) / (MAX_THROTTLE - MIN_THROTTLE)
            done = True
            return REWARD_CRASH - CRASH_SPEED_WEIGHT * norm_throttle, done

        # 1 per timesteps + throttle
        throttle_reward = THROTTLE_REWARD_WEIGHT * (self.last_throttle / MAX_THROTTLE)
        return 1 + throttle_reward, done
