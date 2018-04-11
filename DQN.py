import random
import sys

import tensorflow as tf
import numpy as np
from collections import namedtuple
from lib.envs.bitflip import BitFlipEnv
import os


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
        # processed_state = tf.expand_dims(tf.convert_to_tensor(state), axis=0)
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
            phi = q_approx.process_state(state)
            done = False
            t = 1
            print("\n")
            while not done and t < self.T:
                # for t in range(self.T):
                action = self.execute_policy(sess, phi)
                next_state, reward, done, _ = env.step(action)
                self.replay_buffer.append(Transition(state, action, reward, next_state, done))
                if len(self.replay_buffer) >= batch_size:
                    sample = random.sample(self.replay_buffer, batch_size)
                    phis_batch, action_batch, reward_batch, next_phis_batch, done_batch = \
                        map(np.array, zip(*sample))
                    y_batch = reward_batch + gamma * np.invert(done_batch).astype(np.float32) * \
                              np.amax(q_approx.predict(sess, next_phis_batch), axis=1)
                    loss = q_approx.gradient_step(sess, phis_batch, y_batch, action_batch)
                    print("\rStep {} ({}) @ Episode {}/{}, loss: {}".format(
                        t, current_timestep, episode + 1, self.M, loss), end="")
                    sys.stdout.flush()

                if done:
                    print("Done. Reached reward", reward)

                t += 1
                state = next_state
                phi = q_approx.process_state(state)


tf.reset_default_graph()
np.random.seed(42)

n = 15
goal = np.random.choice([0, 1], size=(n,))
env = BitFlipEnv(n, goal)
q_approx = Q_Approximator(n, n)
dqn = DQN(q_approx, M=10, T=1000)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    dqn.run(sess, env)
    # observation = env.reset()
    # print(q_approx.predict(sess, observation))
