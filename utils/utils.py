# Code adapted from https://github.com/araffin/rl-baselines-zoo
# Author: Antonin Raffin
import glob
import os
import time

import yaml
import tensorflow as tf
import numpy as np
from stable_baselines import logger
from stable_baselines.bench import Monitor
from stable_baselines.common import set_global_seeds
from stable_baselines.common.policies import FeedForwardPolicy as BasePolicy
from stable_baselines.common.policies import register_policy
from stable_baselines.common.vec_env import DummyVecEnv, VecNormalize, \
    VecFrameStack
from stable_baselines.sac.policies import FeedForwardPolicy as SACPolicy
from stable_baselines.ddpg.policies import FeedForwardPolicy as DDPGPolicy

from algos import DDPG, SAC, PPO2
from environment.env import Env

from vae.controller import VAEController
from config import MIN_THROTTLE, MAX_THROTTLE, FRAME_SKIP, N_COMMAND_HISTORY, TEST_FRAME_SKIP

ALGOS = {
    'ddpg': DDPG,
    'sac': SAC,
    'ppo2': PPO2
}

# Used for saving best model
best_mean_reward = -np.inf

# ================== Custom Policies =================

class CustomMlpPolicy(BasePolicy):
    def __init__(self, *args, **kwargs):
        super(CustomMlpPolicy, self).__init__(*args, **kwargs,
                                              layers=[16],
                                              feature_extraction="mlp")


class LargeSACPolicy(SACPolicy):
    def __init__(self, *args, **kwargs):
        super(LargeSACPolicy, self).__init__(*args, **kwargs,
                                              layers=[256, 256],
                                              feature_extraction="mlp")

class TinySACPolicy(SACPolicy):
    def __init__(self, *args, **kwargs):
        super(TinySACPolicy, self).__init__(*args, **kwargs,
                                              layers=[32, 16],
                                              feature_extraction="mlp")

class CustomSACPolicy(SACPolicy):
    def __init__(self, *args, **kwargs):
        super(CustomSACPolicy, self).__init__(*args, **kwargs,
                                              layers=[64, 64],
                                              feature_extraction="mlp")

class CustomDDPGPolicy(DDPGPolicy):
    def __init__(self, *args, **kwargs):
        super(CustomDDPGPolicy, self).__init__(*args, **kwargs,
                                               layers=[32, 8],
                                               feature_extraction="mlp",
                                               layer_norm=True)


register_policy('CustomDDPGPolicy', CustomDDPGPolicy)
register_policy('LargeSACPolicy', LargeSACPolicy)
register_policy('TinySACPolicy', TinySACPolicy)
register_policy('CustomSACPolicy', CustomSACPolicy)
register_policy('CustomMlpPolicy', CustomMlpPolicy)


def load_vae(path=None, z_size=None):
    """
    :param path: (str)
    :param z_size: (int)
    :return: (VAEController)
    """
    # z_size will be recovered from saved model
    if z_size is None:
        assert path is not None

    vae = VAEController(z_size=z_size)
    if path is not None:
        vae.load(path)
    print("Dim VAE = {}".format(vae.z_size))
    return vae


def make_env(client, seed=0, log_dir=None, vae=None, frame_skip=None, n_stack=1):
    """
    Helper function to multiprocess training
    and log the progress.

    :param seed: (int)
    :param log_dir: (str)
    :param vae: (str)
    :param frame_skip: (int)
    :param teleop: (bool)
    """
    if frame_skip is None:
        frame_skip = FRAME_SKIP

    if log_dir is None and log_dir != '':
        log_dir = "/tmp/gym/{}/".format(int(time.time()))
    os.makedirs(log_dir, exist_ok=True)

    def _init():
        set_global_seeds(seed)
        env = Env(client, frame_skip=frame_skip, vae=vae, min_throttle=MIN_THROTTLE,
            max_throttle=MAX_THROTTLE, n_command_history=N_COMMAND_HISTORY,
            n_stack=n_stack)
        env.seed(seed)
        env = Monitor(env, log_dir, allow_early_resets=True)
        return env

    return _init


def create_test_env(client, stats_path=None, seed=0,
                    log_dir='', hyperparams=None):
    """
    Create environment for testing a trained agent

    :param stats_path: (str) path to folder containing saved running averaged
    :param seed: (int) Seed for random number generator
    :param log_dir: (str) Where to log rewards
    :param hyperparams: (dict) Additional hyperparams (ex: n_stack)
    :return: (gym.Env)
    """
    # HACK to save logs
    if log_dir is not None:
        os.environ["OPENAI_LOG_FORMAT"] = 'csv'
        os.environ["OPENAI_LOGDIR"] = os.path.abspath(log_dir)
        os.makedirs(log_dir, exist_ok=True)
        logger.configure()

    vae_path = hyperparams['vae_path']
    if vae_path == '':
        vae_path = os.path.join(stats_path, 'vae.pkl')
    vae = None
    if stats_path is not None and os.path.isfile(vae_path):
        vae = load_vae(vae_path)

    env = DummyVecEnv([make_env(client, seed, log_dir, vae=vae,
                                frame_skip=TEST_FRAME_SKIP)])

    # Load saved stats for normalizing input and rewards
    # And optionally stack frames
    if stats_path is not None:
        if hyperparams['normalize']:
            print("Loading running average")
            print("with params: {}".format(hyperparams['normalize_kwargs']))
            env = VecNormalize(env, training=False, **hyperparams['normalize_kwargs'])
            env.load_running_average(stats_path)

        n_stack = hyperparams.get('n_stack', 0)
        if n_stack > 0:
            print("Stacking {} frames".format(n_stack))
            env = VecFrameStack(env, n_stack)
    return env


def linear_schedule(initial_value):
    """
    Linear learning rate schedule.

    :param initial_value: (float or str)
    :return: (function)
    """
    if isinstance(initial_value, str):
        initial_value = float(initial_value)

    def func(progress):
        """
        Progress will decrease from 1 (beginning) to 0
        :param progress: (float)
        :return: (float)
        """
        return progress * initial_value

    return func


def get_trained_models(log_folder):
    """
    :param log_folder: (str) Root log folder
    :return: (dict) Dict representing the trained agent
    """
    algos = os.listdir(log_folder)
    trained_models = {}
    for algo in algos:
        for env_id in glob.glob('{}/{}/*.pkl'.format(log_folder, algo)):
            # Retrieve env name
            env_id = env_id.split('/')[-1].split('.pkl')[0]
            trained_models['{}-{}'.format(algo, env_id)] = (algo, env_id)
    return trained_models


def get_latest_run_id(log_path, env_id):
    """
    Returns the latest run number for the given log name and log path,
    by finding the greatest number in the directories.

    :param log_path: (str) path to log folder
    :param env_id: (str)
    :return: (int) latest run number
    """
    max_run_id = 0
    for path in glob.glob(log_path + "/{}_[0-9]*".format(env_id)):
        file_name = path.split("/")[-1]
        ext = file_name.split("_")[-1]
        if env_id == "_".join(file_name.split("_")[:-1]) and ext.isdigit() and int(ext) > max_run_id:
            max_run_id = int(ext)
    return max_run_id


def get_saved_hyperparams(stats_path, norm_reward=False):
    """
    :param stats_path: (str)
    :param norm_reward: (bool)
    :return: (dict, str)
    """
    hyperparams = {}
    if not os.path.isdir(stats_path):
        stats_path = None
    else:
        config_file = os.path.join(stats_path, 'config.yml')
        if os.path.isfile(config_file):
            # Load saved hyperparameters
            with open(os.path.join(stats_path, 'config.yml'), 'r') as f:
                hyperparams = yaml.load(f)
            hyperparams['normalize'] = hyperparams.get('normalize', False)
        else:
            obs_rms_path = os.path.join(stats_path, 'obs_rms.pkl')
            hyperparams['normalize'] = os.path.isfile(obs_rms_path)

        # Load normalization params
        if hyperparams['normalize']:
            if isinstance(hyperparams['normalize'], str):
                normalize_kwargs = eval(hyperparams['normalize'])
            else:
                normalize_kwargs = {'norm_obs': hyperparams['normalize'], 'norm_reward': norm_reward}
            hyperparams['normalize_kwargs'] = normalize_kwargs
    return hyperparams, stats_path


def create_callback(algo, save_path, verbose=1):
    """
    Create callback function for saving best model frequently.

    :param algo: (str)
    :param save_path: (str)
    :param verbose: (int)
    :return: (function) the callback function
    """
    if algo != 'sac':
        raise NotImplementedError("Callback creation not implemented yet for {}".format(algo))

    def sac_callback(_locals, _globals):
        """
        Callback for saving best model when using SAC.

        :param _locals: (dict)
        :param _globals: (dict)
        :return: (bool) If False: stop training
        """
        global best_mean_reward
        episode_rewards = _locals['episode_rewards']
        if len(episode_rewards[-101:-1]) == 0:
            return True
        else:
            mean_reward = round(float(np.mean(episode_rewards[-101:-1])), 1)
        if mean_reward > best_mean_reward:
            if verbose >= 1:
                print("Saving best model")
            _locals['self'].save(save_path)
            best_mean_reward = mean_reward

        return True
    return sac_callback
