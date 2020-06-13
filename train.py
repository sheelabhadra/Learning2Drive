# Code adapted from https://github.com/araffin/learning-to-drive-in-5-minutes/
# Author: Sheelabhadra Dey
import argparse
import os
import time
from collections import OrderedDict
from pprint import pprint

import numpy as np
import yaml
from stable_baselines.common import set_global_seeds
from stable_baselines.common.vec_env import VecFrameStack, VecNormalize, DummyVecEnv
from stable_baselines.ddpg import AdaptiveParamNoiseSpec, NormalActionNoise, OrnsteinUhlenbeckActionNoise
from stable_baselines.ppo2.ppo2 import constfn

from config import MIN_THROTTLE, MAX_THROTTLE, FRAME_SKIP,\
    SIM_PARAMS, N_COMMAND_HISTORY, BASE_ENV, ENV_ID, MAX_STEERING_DIFF
from utils.utils import make_env, ALGOS, linear_schedule, get_latest_run_id, load_vae, create_callback

from environment.env import Env
from environment.carla.client import make_carla_client, CarlaClient
from environment.carla.tcp import TCPConnectionError

parser = argparse.ArgumentParser()
parser.add_argument('-tb', '--tensorboard-log', help='Tensorboard log dir', default='', type=str)
parser.add_argument('-i', '--trained-agent', help='Path to a pretrained agent to continue training',
                    default='', type=str)
parser.add_argument('--algo', help='RL Algorithm', default='sac',
                    type=str, required=False, choices=list(ALGOS.keys()))
parser.add_argument('-n', '--n-timesteps', help='Overwrite the number of timesteps', default=-1,
                    type=int)
parser.add_argument('--log-interval', help='Override log interval (default: -1, no change)', default=-1,
                    type=int)
parser.add_argument('-f', '--log-folder', help='Log folder', type=str, default='logs')
parser.add_argument('-vae', '--vae-path', help='Path to saved VAE', type=str, default='')
parser.add_argument('--save-vae', action='store_true', default=False,
                    help='Save VAE')
parser.add_argument('--seed', help='Random generator seed', type=int, default=0)
args = parser.parse_args()

set_global_seeds(args.seed)

tensorboard_log = None if args.tensorboard_log == '' else args.tensorboard_log + '/' + ENV_ID

print("=" * 10, ENV_ID, args.algo, "=" * 10)

vae = None
if args.vae_path != '':
    print("Loading VAE ...")
    vae = load_vae(args.vae_path)

# Load hyperparameters from yaml file
with open('hyperparams/{}.yml'.format(args.algo), 'r') as f:
    hyperparams = yaml.load(f)[BASE_ENV]

# Sort hyperparams that will be saved
saved_hyperparams = OrderedDict([(key, hyperparams[key]) for key in sorted(hyperparams.keys())])
# save vae path
saved_hyperparams['vae_path'] = args.vae_path
if vae is not None:
    saved_hyperparams['z_size'] = vae.z_size

# Save simulation params
for key in SIM_PARAMS:
    saved_hyperparams[key] = eval(key)
pprint(saved_hyperparams)

# Compute and create log path
log_path = os.path.join(args.log_folder, args.algo)
save_path = os.path.join(log_path, "{}_{}".format(ENV_ID, get_latest_run_id(log_path, ENV_ID) + 1))
params_path = os.path.join(save_path, ENV_ID)
os.makedirs(params_path, exist_ok=True)

# Create learning rate schedules for ppo2 and sac
if args.algo in ["ppo2", "sac"]:
    for key in ['learning_rate', 'cliprange']:
        if key not in hyperparams:
            continue
        if isinstance(hyperparams[key], str):
            schedule, initial_value = hyperparams[key].split('_')
            initial_value = float(initial_value)
            hyperparams[key] = linear_schedule(initial_value)
        elif isinstance(hyperparams[key], float):
            hyperparams[key] = constfn(hyperparams[key])
        else:
            raise ValueError('Invalid valid for {}: {}'.format(key, hyperparams[key]))

if args.n_timesteps > 0:
    n_timesteps = args.n_timesteps
else:
    n_timesteps = int(hyperparams['n_timesteps'])
del hyperparams['n_timesteps']

with make_carla_client('localhost', 2000) as client:
    print("CarlaClient connected")

    env = DummyVecEnv([make_env(client, args.seed, vae=vae)])

    # Optional Frame-stacking
    n_stack = 1
    if hyperparams.get('frame_stack', False):
        n_stack = hyperparams['frame_stack']
        env = VecFrameStack(env, n_stack)
        print("Stacking {} frames".format(n_stack))
        del hyperparams['frame_stack']

    # Parse noise string for DDPG
    if args.algo == 'ddpg' and hyperparams.get('noise_type') is not None:
        noise_type = hyperparams['noise_type'].strip()
        noise_std = hyperparams['noise_std']
        n_actions = env.action_space.shape[0]
        if 'adaptive-param' in noise_type:
            hyperparams['param_noise'] = AdaptiveParamNoiseSpec(initial_stddev=noise_std,
                                                                desired_action_stddev=noise_std)
        elif 'normal' in noise_type:
            hyperparams['action_noise'] = NormalActionNoise(mean=np.zeros(n_actions),
                                                            sigma=noise_std * np.ones(n_actions))
        elif 'ornstein-uhlenbeck' in noise_type:
            hyperparams['action_noise'] = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions),
                                                                       sigma=noise_std * np.ones(n_actions))
        else:
            raise RuntimeError('Unknown noise type "{}"'.format(noise_type))
        print("Applying {} noise with std {}".format(noise_type, noise_std))
        del hyperparams['noise_type']
        del hyperparams['noise_std']

    # Train an agent from scratch
    model = ALGOS[args.algo](env=env, tensorboard_log=tensorboard_log, verbose=1, **hyperparams)

    kwargs = {}
    if args.log_interval > -1:
        kwargs = {'log_interval': args.log_interval}

    if args.algo == 'sac':
        kwargs.update({'callback': create_callback(args.algo,
                                                   os.path.join(save_path, ENV_ID + "_best"),
                                                   verbose=1)})

    model.learn(n_timesteps, **kwargs)

    # Save trained model
    model.save(os.path.join(save_path, ENV_ID))
    # Save hyperparams
    with open(os.path.join(params_path, 'config.yml'), 'w') as f:
        yaml.dump(saved_hyperparams, f)

    if args.save_vae and vae is not None:
        print("Saving VAE")
        vae.save(os.path.join(params_path, 'vae'))