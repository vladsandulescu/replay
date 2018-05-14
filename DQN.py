import multiprocessing
import pickle
import random
import sys
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import config
from pathlib import Path
from collections import namedtuple
from joblib import Parallel, delayed
from lib.envs.bitflip import BitFlipEnv
from QApproximator import QApproximator
from QApproximator import QTargetNetworkCopier

np.random.seed(42)
Transition = namedtuple("Transition", ["state", "action", "reward", "next_state", "done"])


# From https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf
class DQN:
    def __init__(self, env, q_approx, q_target_network, epochs, cycles, episodes, episode_timesteps,
                 optimization_steps, minibatch_size, gamma, q_target_network_decay, experience_replay_size,
                 experience_replay_file_suffix):
        self.env = env
        self.q_approx = q_approx
        self.q_target_network = q_target_network
        self.epochs = epochs
        self.cycles = cycles
        self.episodes = episodes
        self.episode_timesteps = episode_timesteps
        self.optimization_steps = optimization_steps
        self.minibatch_size = minibatch_size
        self.gamma = gamma
        self.q_target_network_decay = q_target_network_decay
        self.experience_replay_size = (int)(experience_replay_size)
        self.experience_replay_file = 'experiments/dqn_experience_replay_bits_{}.pkl'.format(
            experience_replay_file_suffix)
        self.experience_replay = []
        self.model_copier = QTargetNetworkCopier(q_approx, q_target_network, q_target_network_decay)

    def run(self, sess):
        self.init_experience_replay(sess)

        for epoch in range(self.epochs):
            # for cycle in range(self.cycles):
            for episode in range(self.episodes):
                state, _ = self.env.reset()
                for t in range(self.episode_timesteps):
                    action = self.q_approx.execute_policy(sess, state)
                    next_state, reward, done, _ = self.env.step(action)
                    self.experience_replay.pop(0)
                    self.experience_replay.append(Transition(state, action, reward, next_state, done))
                    state = next_state
                    if done:
                        print("\n Done. Episode {}/{} @ Reward: {}".format(
                            episode + 1, self.episodes, reward))
                        sys.stdout.flush()
                        return 1

            # for optimization_step in range(self.optimization_steps):
                    sample = random.sample(self.experience_replay, self.minibatch_size)
                    states_batch, action_batch, reward_batch, next_states_batch, done_batch = \
                        map(np.array, zip(*sample))
                    y_batch = reward_batch + self.gamma * np.invert(done_batch).astype(np.float32) * \
                              np.amax(self.q_target_network.predict(sess, next_states_batch), axis=1)
                    loss = self.q_approx.gradient_step(sess, states_batch, y_batch, action_batch)
                    print("\r loss: {}".format(loss), end="")

                    # self.model_copier.run(sess)

        return 0

    def init_experience_replay(self, sess, load_from_disk=True):
        if load_from_disk and Path(self.experience_replay_file).is_file():
            self.experience_replay = pickle.load(open(self.experience_replay_file, 'rb'))
        else:
            state, _ = self.env.reset()
            for t in range(self.experience_replay_size):
                action = self.q_approx.execute_policy(sess, state)
                next_state, reward, done, _ = self.env.step(action)
                self.experience_replay.append(Transition(state, action, reward, next_state, done))
                if done:
                    state, _ = self.env.reset()
                else:
                    state = next_state
            pickle.dump(self.experience_replay, open(self.experience_replay_file, 'wb'))


def run_dqn(n_max=10, single_n=False):
    results = []
    for n in range(1, n_max + 1):
        if single_n and n != n_max:
            continue
        print("\n n =", n)
        tf.reset_default_graph()

        env = BitFlipEnv(n)
        q_approx = QApproximator(n, n, config.PARAMS['batch_size'], scope="approximator")
        q_target_network = QApproximator(n, n, config.PARAMS['batch_size'], scope="target_network")
        dqn = DQN(env, q_approx, q_target_network, epochs=config.PARAMS['n_epochs'], cycles=config.PARAMS['n_cycles'],
                  episodes=config.PARAMS['n_episodes'], episode_timesteps=n,
                  optimization_steps=config.PARAMS['n_optimization_steps'], minibatch_size=config.PARAMS['batch_size'],
                  gamma=config.PARAMS['gamma'], q_target_network_decay=config.PARAMS['polyak_tau'],
                  experience_replay_size=config.PARAMS['experience_replay_size'], experience_replay_file_suffix=n)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            success = dqn.run(sess)
            results.append(Result(n, success))

    results = pd.DataFrame(results)
    results.to_csv('experiments/results_dqn.csv')


Result = namedtuple("Result", field_names=['n', 'success'])


def run_dqn_worker(n):
    results = []
    tf.reset_default_graph()
    print("\n git n =", n)

    env = BitFlipEnv(n)
    q_approx = QApproximator(n, n, config.PARAMS['batch_size'], scope="approximator")
    q_target_network = QApproximator(n, n, config.PARAMS['batch_size'], scope="target_network")
    dqn = DQN(env, q_approx, q_target_network, epochs=config.PARAMS['n_epochs'], cycles=config.PARAMS['n_cycles'],
              episodes=config.PARAMS['n_episodes'], episode_timesteps=n,
              optimization_steps=config.PARAMS['n_optimization_steps'], minibatch_size=config.PARAMS['batch_size'],
              gamma=config.PARAMS['gamma'], q_target_network_decay=config.PARAMS['polyak_tau'],
              experience_replay_size=config.PARAMS['experience_replay_size'], experience_replay_file_suffix=n)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        success = dqn.run(sess)
        results.append(Result(n, success))
    return results


def run_dqn_parallel(n_max=10):
    n_values = range(1, n_max + 1)
    new_results = Parallel(n_jobs=(int)(multiprocessing.cpu_count() / 2 - 1))(
        delayed(run_dqn_worker)(n) for n in list(n_values))
    new_results_flat = [item for result in new_results for item in result]
    results = pd.DataFrame(new_results_flat)
    results.to_csv('experiments/results_dqn_parallel.csv')


def plot(success_rate):
    sns.set_style("whitegrid")
    sns.set_context("paper")
    plt.plot(success_rate.index, success_rate.values, linestyle='--', color='red', label='DQN')
    plt.xticks(np.arange(1, max(success_rate.index) + 1, 1.0))
    plt.yticks(np.arange(0, 1.2, 0.2))
    plt.legend(loc=1, bbox_to_anchor=(0.5, 1.1), ncol=2)
    plt.xlabel('bits')
    plt.ylabel('success rate')
    sns.despine()
    plt.show()


run_dqn_parallel(50)
# run_dqn(38, single_n=True)
results = pd.DataFrame.from_csv('experiments/results_dqn.csv')
success_rate = results.groupby('n')['success'].mean()
plot(success_rate)
