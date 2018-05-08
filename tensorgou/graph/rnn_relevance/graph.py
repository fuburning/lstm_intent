"""
Tensorgou cnn_dssm module graph define
"""

import tensorflow as tf

from tensorflow.python.ops import rnn_cell

__author__ = "Yuanpeng Zhang"
__date__   = "$2016-11-08$"


class tgGraph(object):
    def __init__(self
                 , hiddensize
                 , batchsize
                 , numsteps
                 , keep_prob
                 , numlayers
                 , embeddingnumpy
                 , lr=0.0015
                 , momentum=0.85
                 , istrain=True
                 , distributed=False
                 , global_step=None):

        self.batchsize = batchsize
        self.numsteps = numsteps
        size = hiddensize
        self._input_data = tf.placeholder(tf.int32, [3 * batchsize, numsteps])
        self._lengths = tf.placeholder(tf.int32, [3 * batchsize])

        # Slightly better results can be obtained with forget gate biases
        # initialized to 1 but the hyperparameters of the model would need to be
        # different than reported in the paper.
        lstm_cell = rnn_cell.BasicLSTMCell(size, forget_bias=0.0)
        if istrain and keep_prob < 1:
            lstm_cell = rnn_cell.DropoutWrapper(lstm_cell, output_keep_prob=keep_prob)

        cell = rnn_cell.MultiRNNCell([lstm_cell] * numlayers)

        self._initial_state = cell.zero_state(3 * batchsize, tf.float32)

        offsets = (self._lengths - 1) * (3 * self.batchsize) + tf.range(0, 3 * self.batchsize)

        with tf.device("/cpu:0"):
            self.embedding = tf.get_variable("embedding", initializer=tf.constant(embeddingnumpy))

        inputs = tf.nn.embedding_lookup(self.embedding, self._input_data)
        if istrain and keep_prob < 1:
            inputs = tf.nn.dropout(inputs, keep_prob)

        # Simplified version of tensorflow.models.rnn.rnn.py's rnn().
        # This builds an unrolled LSTM for tutorial purposes only.
        # In general, use the rnn() or state_saving_rnn() from rnn.py.
        #
        # The alternative version of the code below is:
        #
        # from tensorflow.models.rnn import rnn
        # inputs = [tf.squeeze(input_, [1])
        #           for input_ in tf.split(1, num_steps, inputs)]
        # outputs, states = rnn.rnn(cell, inputs, initial_state=self._initial_state)
        outputs = []
        states = []
        state = self._initial_state
        with tf.variable_scope("RNN"):
            for time_step in range(numsteps):
                if time_step > 0: tf.get_variable_scope().reuse_variables()
                (cell_output, state) = cell(inputs[:, time_step, :], state)
                outputs.append(cell_output)
                states.append(state)

        output = tf.reshape(tf.concat(0, outputs), [-1, size])
        output = tf.gather(output, offsets)
        logits = tf.nn.xw_plus_b(output,
                                 tf.get_variable("softmax_w", [size, hiddensize]),
                                 tf.get_variable("softmax_b", [hiddensize]))

        logists_splits = tf.split(0, 3, logits)

        norm0 = tf.sqrt(tf.reduce_sum(tf.square(logists_splits[0]), 1, keep_dims = True))
        norm1 = tf.sqrt(tf.reduce_sum(tf.square(logists_splits[1]), 1, keep_dims = True))
        norm2 = tf.sqrt(tf.reduce_sum(tf.square(logists_splits[2]), 1, keep_dims = True))

        cosine01 = tf.reduce_sum(tf.mul(logists_splits[0], logists_splits[1]), 1, keep_dims = True) / (tf.mul(norm0, norm1))
        cosine02 = tf.reduce_sum(tf.mul(logists_splits[0], logists_splits[2]), 1, keep_dims = True) / (tf.mul(norm0, norm2))

        self._cost = tf.reduce_sum(tf.log(1.0 + tf.exp(-10.0 * (cosine01 - cosine02))))

        diff = cosine01 - cosine02
        correct_prediction = (tf.sign(diff) + 1) / 2
        self._corrects_num = tf.reduce_sum(tf.cast(correct_prediction, "float"))

        if istrain is True:
            #optimizer = tf.train.RMSPropOptimizer(learning_rate = config.lr, decay = config.decay, momentum = config.momentum, epsilon = config.epsilon)
            optimizer = tf.train.MomentumOptimizer(learning_rate = lr, momentum = momentum)
            self.train_op_ = optimizer.minimize(self._cost, global_step=global_step)
            if distributed is True:
                tf.scalar_summary("cost", self._cost)
                tf.scalar_summary("accuracy", self._corrects_num)
                self.summary_op = tf.merge_all_summaries()