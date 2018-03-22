import tensorflow as tf
import numpy as np
from collections import namedtuple

from lib.envs.bitflip import BitFlipEnv


class DQN:
    '''
    DQN with experience replay
    '''

    def __init__(self, N=1e3, M=1e03, T=200):
        self.N = N
        self.M = M
        self.T = T

    def create_layer(self, input, batch_size, units):
        with tf.variable_scope("layer"):
            weights = tf.Variable(tf.random_normal([batch_size] + units, stddev=0.35), name="weights")
            biases = tf.Variable(tf.zeros(units), name="biases")
            output = tf.nn.relu(tf.add(tf.matmul(input, weights), biases))
        return output

    def create_tf_model(self, batch_size: int, state_shape: int, num_actions: int):
        global_step = tf.Variable(0, name='global_step', trainable=False)

        with tf.variable_scope("DQN"):
            with tf.variable_scope("Q_input"):
                self.x = tf.placeholder(dtype=tf.float32, shape=[batch_size] + state_shape, name="x_batch")

            with tf.variable_scope("Q_layers"):
                self.layer1 = self.create_layer(self.x, batch_size, 256)
                self.predictions = tf.layers.dense(self.layer1, units=num_actions, name="predictions")

            with tf.variable_scope("Q_output"):
                self.actions = tf.placeholder(dtype=tf.float32, shape=[batch_size], name="actions")
                self.y = tf.placeholder(dtype=tf.float32, shape=[batch_size], name="y")

            with tf.variable_scope("Q_losses"):
                self.loss = tf.reduce_mean(tf.squared_difference(self.y, self.predictions))
                self.optimizer = tf.train.RMSPropOptimizer(learning_rate=0.00025, momentum=0.95, epsilon=0.01)
                self.train_op = self.optimizer.minimize(self.loss, global_step=global_step)

            self.summaries = tf.summary.merge([
                tf.summary.scalar("loss", self.loss),
                tf.summary.histogram("predictions", self.predictions)
            ])

    def predict(self, sess: tf.Session, state):
        return sess.run(self.predictions, feed_dict={self.x: state})

    def gradient_step(self, sess: tf.Session, states_batch, y_batch, actions_batch):
        self.summaries, self.loss, _ = sess.run(fetches=[self.summaries, self.loss, self.train_op],
                                                feed_dict={self.x: states_batch, self.y: y_batch,
                                                           self.actions: actions_batch})

    def execute_policy(self, sess, state, epsilon=0.1):
        q_function = self.predict(sess, state)
        explore = np.random.choice([0, 1], p=[1 - epsilon, epsilon])
        if explore:
            action = np.random.choice(range(q_function))
        else:
            action = np.argmax(q_function)
        return action

    def run(self):
        gamma = 0.9
        batch_size = 10000
        n = 10
        np.random.seed(42)
        goal = np.random.choice([0, 1], size=(n,))

        env = BitFlipEnv(n, goal)
        tf.reset_default_graph()
        Transition = namedtuple("Transition", ["state", "action", "reward", "next_state", "done"])
        self.replay_buffer = []
        self.create_tf_model(batch_size, n, n)

        with tf.Session() as sess:
            sess.run(tf.initialize_all_variables())
            for episode in range(self.M):
                state = env.reset()
                for t in range(self.T):
                    action = self.execute_policy(sess, state)
                    next_state, reward, done, _ = env.step(action)
                    self.replay_buffer.append(Transition(state, action, reward, next_state, done))
                    sample = np.random.sample(self.replay_buffer, n=batch_size)
                    states_batch, action_batch, reward_batch, next_states_batch, done_batch = \
                        map(np.array, zip(*sample))
                    y_batch = reward_batch + gamma * np.invert(done_batch).astype(np.float32) * \
                              np.amax(self.predict(sess, next_states_batch), axis=1)
                    self.gradient_step(sess, states_batch, y_batch, action_batch)
