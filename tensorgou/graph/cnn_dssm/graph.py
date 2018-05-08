"""
Tensorgou cnn_dssm module graph define
"""

import tensorflow as tf

from tensorflow.python.ops import rnn_cell
from tensorflow.python.ops import seq2seq

__author__ = "Yuanpeng Zhang"
__date__   = "$2016-10-18$"


class tgGraph(object):
    def __init__(self
                 , modelsize
                 , batchsize
                 , numsteps
                 , embeddingnumpy
                 , embeddingsize=100
                 , lr=0.0015
                 , momentum=0.85
                 , istrain=True
                 , distributed=False
                 , global_step=None):
        self.inputdata = []
        for i in range(3):
            self.inputdata.append(tf.placeholder(tf.int32, [3 * batchsize, numsteps]))

        """
        self.embeddingnumpy = embeddingnumpy
        assert len(self.embeddingnumpy[0]) == embeddingsize
        with tf.device('/cpu:0'):
            self.embedding = tf.get_variable("embedding"
                                            , [len(self.embeddingnumpy), embeddingsize]
                                            , initializer=initializer)
        """
        with tf.device('/cpu:0'):
            assert len(embeddingnumpy[0]) == embeddingsize
            self.embedding = tf.get_variable("embedding", initializer=tf.constant(embeddingnumpy))

        inputsarr = []
        for i in range(3):
            inputsarr.append(tf.nn.embedding_lookup(self.embedding, self.inputdata[i]))

        inputs = tf.concat(2, [inputsarr[0], inputsarr[1], inputsarr[2]])
        inputs_flatten = tf.reshape(inputs, [-1, 3 * embeddingsize])
        inputs_trans = tf.transpose(inputs_flatten)

        conv_W = tf.get_variable("conv_W", shape = [modelsize[0], 3 * embeddingsize])
        conved = tf.matmul(conv_W, inputs_trans)
        sub_matrix_list = tf.split(1, 3 * batchsize, conved)
        packed_sub_matrxes = tf.pack(sub_matrix_list)
        pooled_input = tf.map_fn(lambda x : tf.reduce_max(x, 1), packed_sub_matrxes, back_prop = True)

        layer_id = 0
        DNN_W_0 = tf.get_variable("DNN_W_0"
                                  , shape = [modelsize[layer_id], modelsize[layer_id + 1]])
        DNN_B_0 = tf.get_variable("DNN_B_0"
                                  , shape = [modelsize[layer_id + 1]])
        activation_0 = tf.nn.relu(tf.matmul(pooled_input, DNN_W_0) + DNN_B_0)

        layer_id = 1
        DNN_W_1 = tf.get_variable("DNN_W_1"
                                  , shape = [modelsize[layer_id], modelsize[layer_id + 1]])
        DNN_B_1 = tf.get_variable("DNN_B_1"
                                  , shape = [modelsize[layer_id + 1]])
        activation_1 = tf.nn.relu(tf.matmul(activation_0, DNN_W_1) + DNN_B_1)

        layer_id = 2
        DNN_W_2 = tf.get_variable("DNN_W_2"
                                  , shape = [modelsize[layer_id], modelsize[layer_id + 1]])
        DNN_B_2 = tf.get_variable("DNN_B_2"
                                  , shape = [modelsize[layer_id + 1]])
        activation_2 = tf.matmul(activation_1, DNN_W_2) + DNN_B_2

        logists_splits = tf.split(0, 3, activation_2)

        norm0 = tf.sqrt(tf.reduce_sum(tf.square(logists_splits[0]), 1, keep_dims = True))
        norm1 = tf.sqrt(tf.reduce_sum(tf.square(logists_splits[1]), 1, keep_dims = True))
        norm2 = tf.sqrt(tf.reduce_sum(tf.square(logists_splits[2]), 1, keep_dims = True))

        cosine01 = tf.reduce_sum(tf.mul(logists_splits[0], logists_splits[1])
                                 , 1
                                 , keep_dims = True) / (tf.mul(norm0, norm1))
        cosine02 = tf.reduce_sum(tf.mul(logists_splits[0], logists_splits[2])
                                 , 1
                                 , keep_dims = True) / (tf.mul(norm0, norm2))

        self._cost = tf.reduce_sum(tf.log(1.0 + tf.exp(-10.0 * (cosine01 - cosine02))))

        diff = cosine01 - cosine02
        correct_prediction = (tf.sign(diff) + 1) / 2
        self._corrects_num = tf.reduce_sum(tf.cast(correct_prediction, "float"))

        if istrain is True:
            optimizer = tf.train.MomentumOptimizer(learning_rate = lr, momentum = momentum)
            self.train_op_ = optimizer.minimize(self._cost, global_step=global_step)

            if distributed is True:
                tf.scalar_summary("cost", self._cost)
                tf.scalar_summary("accuracy", self._corrects_num)
                self.summary_op = tf.merge_all_summaries()
