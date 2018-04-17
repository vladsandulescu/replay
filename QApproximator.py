import os
import tensorflow as tf
import numpy as np

np.random.seed(42)


class QApproximator:
    def __init__(self, states, num_actions, batch_size, scope="estimator", experiments_dir='experiments'):
        self.summary_writer = None
        self.create_tf_model(states, num_actions, batch_size)
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

    def create_tf_model(self, states, num_actions, batch_size):
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

    def execute_policy(self, sess, state, epsilon=0.1):
        q_function = self.predict(sess, state)
        explore = np.random.choice([0, 1], p=[1 - epsilon, epsilon])
        if explore:
            action = np.random.randint(0, len(q_function[0]))
        else:
            action = np.argmax(q_function)
        return action
