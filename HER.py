import multiprocessing
import pickle
import random
import sys
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import namedtuple
from joblib import Parallel, delayed
from lib.envs.bitflip import BitFlipEnv
from QApproximator import QApproximator
from QApproximator import QTargetNetworkCopier

np.random.seed(42)
Transition = namedtuple("Transition", ["state", "action", "reward", "next_state", "done"])


# From https://arxiv.org/pdf/1707.01495.pdf
class HER:
    def __init__(self, env, q_approx, q_target_network, epochs, cycles, episodes, episode_timesteps,
                 optimization_steps, minibatch_size, gamma, q_target_network_decay, experience_replay_size):
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
        self.experience_replay = []
        self.model_copier = QTargetNetworkCopier(q_approx, q_target_network, q_target_network_decay)

    def run(self, sess):
        self.init_experience_replay(sess)

        for epoch in range(self.epochs):
            for cycle in range(self.cycles):
                for episode in range(self.episodes):
                    episode_experience = []
                    state, goal = self.env.reset()
                    for t in range(self.episode_timesteps):
                        state_concat_goal = np.concatenate([state, goal], axis=-1)
                        action = self.q_approx.execute_policy(sess, state_concat_goal)
                        next_state, reward, done, _ = self.env.step(action)
                        episode_experience.append(Transition(state, action, reward, next_state, done))

                        if done:
                            print("\n Done. Episode {}/{} @ cycle {}/{} @ epoch {}/{}. Reward: {}".format(
                                episode + 1, self.episodes, cycle + 1, self.cycles, epoch + 1, self.epochs, reward))
                            sys.stdout.flush()
                            return 1

                        state = next_state

                    for t in range(self.episode_timesteps):
                        state, action, reward, next_state, done = episode_experience[t]
                        state_concat_goal = np.concatenate([state, goal], axis=-1)
                        next_state_concat_goal = np.concatenate([next_state, goal], axis=-1)
                        self.experience_replay.pop(0)
                        self.experience_replay.append(
                            Transition(state_concat_goal, action, reward, next_state_concat_goal, done))
                        additional_goals_sample = len(episode_experience) - 1
                        _, _, _, new_goal, _ = episode_experience[additional_goals_sample]
                        state_concat_new_goal = np.concatenate([state, new_goal], axis=-1)
                        next_state_concat_new_goal = np.concatenate([next_state, new_goal], axis=-1)
                        self.experience_replay.pop(0)
                        self.experience_replay.append(
                            Transition(state_concat_new_goal, action, reward, next_state_concat_new_goal, done))

                for optimization_step in range(self.optimization_steps):
                    sample = random.sample(self.experience_replay, self.minibatch_size)
                    states_batch, action_batch, reward_batch, next_states_batch, done_batch = \
                        map(np.array, zip(*sample))
                    y_batch = reward_batch + self.gamma * np.invert(done_batch).astype(np.float32) * \
                              np.amax(self.q_approx.predict(sess, next_states_batch), axis=1)
                    loss = self.q_approx.gradient_step(sess, states_batch, y_batch, action_batch)
                    print("\r Optimization step {}/{} @ cycle {}/{} @ epoch {}/{}, loss: {}".format(
                        optimization_step + 1, self.optimization_steps, cycle + 1, self.cycles, epoch + 1, self.epochs,
                        loss), end="")

                self.model_copier.run(sess)

        return 0

    def init_experience_replay(self, sess, load_from_disk=True):
        experience_replay_file = 'experiments/her_experience_replay.pkl'
        if load_from_disk and Path(experience_replay_file).is_file():
            self.experience_replay = pickle.load(open(experience_replay_file, 'rb'))
        else:
            state, goal = self.env.reset()
            for t in range(self.experience_replay_size):
                state_concat_goal = np.concatenate([state, goal], axis=-1)
                action = self.q_approx.execute_policy(sess, state_concat_goal)
                next_state, reward, done, _ = self.env.step(action)
                next_state_concat_goal = np.concatenate([next_state, goal], axis=-1)
                self.experience_replay.append(
                    Transition(state_concat_goal, action, reward, next_state_concat_goal, done))
                if done:
                    state, goal = self.env.reset()
                else:
                    state = next_state
            pickle.dump(self.experience_replay, open(experience_replay_file, 'wb'))


def run_her(n_max=10):
    Result = namedtuple("Result", field_names=['n', 'success'])
    results = []

    for n in range(1, n_max + 1):
        print("\n n =", n)
        tf.reset_default_graph()

        env = BitFlipEnv(n)
        batch_size = 128
        q_approx = QApproximator(n * 2, n, batch_size, scope="approximator")
        q_target_network = QApproximator(n * 2, n, batch_size, scope="target_network")
        her = HER(env, q_approx, q_target_network, epochs=200, cycles=50, episodes=16, episode_timesteps=n,
                  optimization_steps=40, minibatch_size=batch_size, gamma=0.98, q_target_network_decay=0.05,
                  experience_replay_size=1e6)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            success = her.run(sess)
            results.append(Result(n, success))

    results = pd.DataFrame(results)
    results.to_csv('experiments/results_her.csv')


Result = namedtuple("Result", field_names=['n', 'success'])


def run_her_worker(n):
    results = []
    print("n =", n)
    tf.reset_default_graph()

    env = BitFlipEnv(n)
    batch_size = 128
    q_approx = QApproximator(n, n, batch_size, scope="approximator")
    q_target_network = QApproximator(n, n, batch_size, scope="target_network")
    her = HER(env, q_approx, q_target_network, epochs=200, cycles=50, episodes=16, episode_timesteps=n,
              optimization_steps=40, minibatch_size=batch_size, gamma=0.98, q_target_network_decay=0.05,
              experience_replay_size=1e6)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        success = her.run(sess)
        results.append(Result(n, success))
    return results


def run_her_parallel(n_max=10):
    n_values = range(1, n_max + 1)
    new_results = Parallel(n_jobs=(int)(multiprocessing.cpu_count() / 2))(
        delayed(run_her_worker)(n) for n in list(n_values))
    new_results_flat = [item for result in new_results for item in result]
    results = pd.DataFrame(new_results_flat)
    results.to_csv('experiments/results_her_parallel.csv')


def plot(success_rate):
    sns.set_style("whitegrid")
    sns.set_context("paper")
    plt.plot(success_rate.index, success_rate.values, linestyle='-', color='blue', label='DQN + HER')
    plt.xticks(np.arange(1, max(success_rate.index) + 1, 1.0))
    plt.yticks(np.arange(0, 1.2, 0.2))
    plt.legend(loc=1, bbox_to_anchor=(0.5, 1.1), ncol=2)
    plt.xlabel('bits')
    plt.ylabel('success rate')
    sns.despine()
    plt.show()


run_her_parallel(50)
results = pd.DataFrame.from_csv('experiments/results_her_parallel.csv')
success_rate = results.groupby('n')['success'].mean()
plot(success_rate)
