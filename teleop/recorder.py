import os

import cv2


class Recorder(object):
    """
    Class to record images for offline VAE training

    :param env: (Gym env)
    :param folder: (str)
    :param start_recording: (bool)
    :param verbose: (int)
    """

    def __init__(self, env, folder='logs/recorded_data/', start_recording=False, verbose=0):
        super(Recorder, self).__init__()
        self.env = env
        self.is_recording = start_recording
        self.folder = folder
        self.current_idx = 0
        self.verbose = verbose
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        # Create folder if needed
        os.makedirs(folder, exist_ok=True)

        images_idx = [int(im.split('.jpg')[0]) for im in os.listdir(folder) if im.endswith('.jpg')]
        if len(images_idx) > 0:
            self.current_idx = max(images_idx)

        if verbose >= 1:
            print("Recorder current idx: {}".format(self.current_idx))

    def reset(self):
        obs = self.env.reset()
        if self.is_recording:
            self.save_image()
        return obs

    def render(self, mode='rgb_array'):
        return self.env.render(mode=mode)

    def seed(self, seed=None):
        return self.env.seed(seed)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        if self.is_recording:
            self.save_image()
        return obs, reward, done, info

    def save_image(self):
        image = self.env.render(mode='rgb_array')
        # Convert RGB to BGR
        image = image[:, :, ::-1]
        cv2.imwrite("{}/{}.jpg".format(self.folder, self.current_idx), image)
        if self.verbose >= 2:
            print("Saving", "{}/{}.jpg".format(self.folder, self.current_idx))
        self.current_idx += 1

    def set_recording_status(self, is_recording):
        self.is_recording = is_recording
        if self.verbose >= 1:
            print("Setting recording to {}".format(is_recording))

    def toggle_recording(self):
        self.set_recording_status(not self.is_recording)

    def exit_scene(self):
        self.env.exit_scene()
