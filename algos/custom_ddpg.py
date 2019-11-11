# Original author: Roma Sokolkov
# Edited by Antonin Raffin
import time

import numpy as np
from mpi4py import MPI

from stable_baselines import logger
from stable_baselines.ddpg.ddpg import DDPG
from stable_baselines.common import TensorboardWriter


class DDPGWithVAE(DDPG):
    """
    Custom DDPG version in order to work with donkey car env.
    It is adapted from the stable-baselines version.

    Changes:
    - optimization is done after each episode
    - more verbosity.
    """
    def learn(self, total_timesteps, callback=None, seed=None,
              log_interval=1, tb_log_name="DDPG", print_freq=100):
        with TensorboardWriter(self.graph, self.tensorboard_log, tb_log_name) as writer:

            rank = MPI.COMM_WORLD.Get_rank()
            # we assume symmetric actions.
            assert np.all(np.abs(self.env.action_space.low) == self.env.action_space.high)

            self.episode_reward = np.zeros((1,))
            with self.sess.as_default(), self.graph.as_default():
                # Prepare everything.
                self._reset()
                episode_reward = 0.
                episode_step = 0
                episodes = 0
                step = 0
                total_steps = 0

                start_time = time.time()

                actor_losses = []
                critic_losses = []
                should_return = False

                while True:
                    obs = self.env.reset()
                    # Rollout one episode.
                    while True:
                        if total_steps >= total_timesteps:
                            if should_return:
                                return self
                            should_return = True
                            break

                        # Predict next action.
                        action, q_value = self._policy(obs, apply_noise=True, compute_q=True)
                        if self.verbose >= 2:
                            print(action)
                        assert action.shape == self.env.action_space.shape

                        # Execute next action.
                        new_obs, reward, done, info = self.env.step(action * np.abs(self.action_space.low))

                        step += 1
                        total_steps += 1
                        if rank == 0 and self.render:
                            self.env.render()
                        episode_reward += reward
                        episode_step += 1

                        if print_freq > 0 and episode_step % print_freq == 0 and episode_step > 0:
                            print("{} steps".format(episode_step))

                        # Book-keeping.
                        self._store_transition(obs, action, reward, new_obs, done)

                        obs = new_obs
                        if callback is not None:
                            # Only stop training if return value is False, not when it is None. This is for backwards
                            # compatibility with callbacks that have no return statement.
                            if callback(locals(), globals()) is False:
                                return self

                        if done:
                            print("Episode finished. Reward: {:.2f} {} Steps".format(episode_reward, episode_step))

                            # Episode done.
                            episode_reward = 0.
                            episode_step = 0
                            episodes += 1

                            self._reset()
                            obs = self.env.reset()
                            # Finish rollout on episode finish.
                            break

                    # Train DDPG.
                    actor_losses = []
                    critic_losses = []
                    train_start = time.time()
                    for t_train in range(self.nb_train_steps):
                        critic_loss, actor_loss = self._train_step(0, None, log=t_train == 0)
                        critic_losses.append(critic_loss)
                        actor_losses.append(actor_loss)
                        self._update_target_net()
                    print("DDPG training duration: {:.2f}s".format(time.time() - train_start))

                    mpi_size = MPI.COMM_WORLD.Get_size()
                    # Log stats.
                    # XXX shouldn't call np.mean on variable length lists
                    duration = time.time() - start_time
                    stats = self._get_stats()
                    combined_stats = stats.copy()
                    combined_stats['train/loss_actor'] = np.mean(actor_losses)
                    combined_stats['train/loss_critic'] = np.mean(critic_losses)
                    combined_stats['total/duration'] = duration
                    combined_stats['total/steps_per_second'] = float(step) / float(duration)
                    combined_stats['total/episodes'] = episodes

                    combined_stats_sums = MPI.COMM_WORLD.allreduce(
                        np.array([as_scalar(x) for x in combined_stats.values()]))
                    combined_stats = {k: v / mpi_size for (k, v) in zip(combined_stats.keys(), combined_stats_sums)}

                    # Total statistics.
                    combined_stats['total/steps'] = step

                    for key in sorted(combined_stats.keys()):
                        logger.record_tabular(key, combined_stats[key])
                    logger.dump_tabular()
                    logger.info('')


def as_scalar(scalar):
    """
    check and return the input if it is a scalar, otherwise raise ValueError

    :param scalar: (Any) the object to check
    :return: (Number) the scalar if x is a scalar
    """
    if isinstance(scalar, np.ndarray):
        assert scalar.size == 1
        return scalar[0]
    elif np.isscalar(scalar):
        return scalar
    else:
        raise ValueError('expected scalar, got %s' % scalar)
