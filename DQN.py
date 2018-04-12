import random
import os
import sys

import tensorflow as tf
import numpy as np
import pandas as pd
from collections import namedtuple
from lib.envs.bitflip import BitFlipEnv
import matplotlib.pyplot as plt
import seaborn as sns

np.random.seed(42)


class Q_Approximator:
    def __init__(self, states, num_actions, scope="estimator", experiments_dir='experiments'):
        self.summary_writer = None
        # with tf.variable_scope(scope):
        self.create_tf_model(states, num_actions)
        if experiments_dir:
            summaries_dir = os.path.join(experiments_dir, "summaries_{}".format(scope))
            if not os.path.exists(summaries_dir):
                os.makedirs(summaries_dir)
            self.summary_writer = tf.summary.FileWriter(summaries_dir)

    def create_layer(self, input, numUnits):
        numInput = input.get_shape()[1].value
        with tf.variable_scope("layer"):
            weights = tf.Variable(tf.random_normal([numInput, numUnits], stddev=0.35), name="weights")
            biases = tf.Variable(tf.zeros(numUnits), name="biases")
            output = tf.nn.relu(tf.add(tf.matmul(input, weights), biases))
        return output

    def create_tf_model(self, states, num_actions, batch_size=1000):
        global_step = tf.Variable(0, name='global_step', trainable=False)

        with tf.variable_scope("DQN"):
            with tf.variable_scope("Q_input"):
                self.x = tf.placeholder(dtype=tf.float32, shape=[None, states], name="x")
                self.y = tf.placeholder(dtype=tf.float32, shape=[None], name="y")
                self.actions = tf.placeholder(dtype=tf.int32, shape=[None], name="actions")

            with tf.variable_scope("Q_layers"):
                self.layer1 = self.create_layer(self.x, 256)
                self.predictions = tf.layers.dense(self.layer1, units=num_actions, name="predictions")

            with tf.variable_scope("Q_output"):
                gather_indices = tf.range(batch_size) * tf.shape(self.predictions)[1] + self.actions
                self.actions_predictions = tf.gather(tf.reshape(self.predictions, [-1]), gather_indices)

            with tf.variable_scope("Q_losses"):
                self.loss = tf.reduce_mean(tf.squared_difference(self.y, self.actions_predictions))
                self.optimizer = tf.train.RMSPropOptimizer(learning_rate=0.00025, momentum=0.95, epsilon=0.01)
                self.train_op = self.optimizer.minimize(self.loss, global_step=global_step)

            self.summaries = tf.summary.merge([
                tf.summary.scalar("loss", self.loss),
                tf.summary.histogram("predictions", self.predictions)
            ])

    def process_state(self, state):
        return np.expand_dims(state, axis=0)

    def predict(self, sess: tf.Session, processed_state):
        return sess.run(self.predictions, feed_dict={self.x: processed_state})

    def gradient_step(self, sess: tf.Session, states_batch, y_batch, actions_batch):
        summaries, global_step, loss, _ = sess.run(
            fetches=[self.summaries, tf.train.get_global_step(), self.loss, self.train_op],
            feed_dict={self.x: states_batch, self.y: y_batch,
                       self.actions: actions_batch})
        if self.summary_writer:
            self.summary_writer.add_summary(summaries, global_step)

        return loss


class DQN:
    def __init__(self, q_approx, M, T):
        self.q_approx = q_approx
        self.M = M
        self.T = T
        self.replay_buffer = []

    def execute_policy(self, sess, state, epsilon=0.1):
        q_function = self.q_approx.predict(sess, state)
        explore = np.random.choice([0, 1], p=[1 - epsilon, epsilon])
        if explore:
            action = np.random.randint(0, len(q_function[0]))
        else:
            action = np.argmax(q_function)
        return action

    def run(self, sess, env, gamma=0.9, batch_size=1000):
        Transition = namedtuple("Transition", ["state", "action", "reward", "next_state", "done"])
        current_timestep = sess.run(tf.train.get_global_step())

        for episode in range(self.M):
            state = env.reset()
            phi = self.q_approx.process_state(state)
            done = False
            t = 1
            print("\n")
            while not done and t < self.T:
                action = self.execute_policy(sess, phi)
                next_state, reward, done, _ = env.step(action)
                self.replay_buffer.append(Transition(state, action, reward, next_state, done))

                # If we are building the initial replay memory and by chance get a 0 reward, then start over
                if len(self.replay_buffer) < batch_size:
                    if done:
                        state = env.reset()
                        phi = self.q_approx.process_state(state)
                        done = False
                else:
                    sample = random.sample(self.replay_buffer, batch_size)
                    phis_batch, action_batch, reward_batch, next_phis_batch, done_batch = \
                        map(np.array, zip(*sample))
                    y_batch = reward_batch + gamma * np.invert(done_batch).astype(np.float32) * \
                              np.amax(self.q_approx.predict(sess, next_phis_batch), axis=1)
                    loss = self.q_approx.gradient_step(sess, phis_batch, y_batch, action_batch)
                    print("\rStep {} ({}) @ Episode {}/{}, loss: {}".format(
                        t, current_timestep, episode + 1, self.M, loss), end="")
                    sys.stdout.flush()

                    if done:
                        print("\nDone. Reward =", reward)
                        return 1

                    t += 1
                    state = next_state
                    phi = self.q_approx.process_state(state)
        return 0


def plot(success_rate):
    sns.set_style("whitegrid")
    sns.set_context("paper")
    plt.plot(success_rate.index, success_rate.values, '-', label='DQN')
    plt.xticks(np.arange(1, max(success_rate.index) + 1, 1.0))
    plt.yticks(np.arange(0, 1.2, 0.2))
    plt.legend(loc=1, bbox_to_anchor=(0.5, 1.1), ncol=2)
    plt.xlabel('bits')
    plt.ylabel('success rate')
    sns.despine()
    plt.show()


def run_dqn(n_max=10):
    Result = namedtuple("Result", field_names=['n', 'trial', 'success'])
    results = []

    for n in range(1, n_max + 1):
        for trial in range(5):
            tf.reset_default_graph()
            goal = np.random.choice([0, 1], size=(n,))
            env = BitFlipEnv(n, goal)
            q_approx = Q_Approximator(n, n)
            dqn = DQN(q_approx, M=20, T=1000)

            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                success = dqn.run(sess, env)
                results.append(Result(n, trial, success))

    results = pd.DataFrame(results)
    results.to_csv('experiments/results.csv')

# run_dqn(10)
results = pd.DataFrame.from_csv('experiments/results.csv')
success_rate = results.groupby('n')['success'].mean()
plot(success_rate)
