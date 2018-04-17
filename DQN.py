import random
import sys

import tensorflow as tf
import numpy as np
import pandas as pd
from collections import namedtuple
from lib.envs.bitflip import BitFlipEnv
import matplotlib.pyplot as plt
import seaborn as sns
from QApproximator import QApproximator

np.random.seed(42)


class DQN:
    def __init__(self, q_approx, n, M, T, B, gamma, replay_buffer_size):
        self.q_approx = q_approx
        self.n = n
        self.M = M
        self.T = T
        self.T = T
        self.B = B
        self.gamma = gamma
        self.replay_buffer_size = (int)(replay_buffer_size)
        self.replay_buffer = []
        self.G = [np.random.choice([0, 1], size=self.n)]
        self.env = BitFlipEnv(n)
        self.Transition = namedtuple("Transition", ["state", "goal", "action", "reward", "next_state", "done"])

    def run(self, sess):
        current_timestep = sess.run(tf.train.get_global_step())
        goal = random.sample(self.G, 1)[0]
        self.init_replay_buffer(sess)

        for episode in range(self.M):
            state = self.env.reset()
            phi = self.q_approx.process_state(state)
            print("\n")
            for t in range(self.T):
                action = self.q_approx.execute_policy(sess, phi)
                next_state, reward, done, _ = self.env.step(action, goal)
                self.replay_buffer.append(self.Transition(state, goal, action, reward, next_state, done))

                if done:
                    print("\nDone. Reward =", reward)
                    return 1

                sample = random.sample(self.replay_buffer, self.B)
                phis_batch, goals_batch, action_batch, reward_batch, next_phis_batch, done_batch = \
                    map(np.array, zip(*sample))
                y_batch = reward_batch + self.gamma * np.invert(done_batch).astype(np.float32) * \
                          np.amax(self.q_approx.predict(sess, next_phis_batch), axis=1)
                loss = self.q_approx.gradient_step(sess, phis_batch, y_batch, action_batch)
                print("\rStep {} ({}) @ Episode {}/{}, loss: {}".format(
                    t, current_timestep, episode + 1, self.M, loss), end="")
                sys.stdout.flush()

                state = next_state
                phi = self.q_approx.process_state(state)

        return 0

    def init_replay_buffer(self, sess):
        goal = random.sample(self.G, 1)[0]
        state = self.env.reset()
        phi = self.q_approx.process_state(state)
        print("\n")
        for t in range(self.replay_buffer_size):
            action = self.q_approx.execute_policy(sess, phi)
            next_state, reward, done, _ = self.env.step(action, goal)
            self.replay_buffer.append(self.Transition(state, goal, action, reward, next_state, done))
            if done:
                goal = random.sample(self.G, 1)[0]
                state = self.env.reset()
                phi = self.q_approx.process_state(state)
            else:
                state = next_state
                phi = self.q_approx.process_state(state)


def run_dqn(n_max=10):
    Result = namedtuple("Result", field_names=['n', 'trial', 'success'])
    results = []

    for n in range(1, n_max + 1):
        for trial in range(5):
            print("n =", n, "trial =", trial)
            tf.reset_default_graph()

            batch_size = 128
            q_approx = QApproximator(n, n, batch_size)
            dqn = DQN(q_approx, n, M=200, T=n, B=batch_size, gamma=0.98, replay_buffer_size=1e3)
            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                success = dqn.run(sess)
                results.append(Result(n, trial, success))

    results = pd.DataFrame(results)
    results.to_csv('experiments/results_dqn.csv')


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


run_dqn(25)
results = pd.DataFrame.from_csv('experiments/results_dqn.csv')
success_rate = results.groupby('n')['success'].mean()
plot(success_rate)
