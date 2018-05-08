"""
Tensorgou bidirectional lstm module graph define
    hiddensize = 256
    embedding_dim = 100
    projectsize = 128
    vocabsize = 25260
    target_size = vocab_size
"""

import tensorflow as tf

from tensorflow.python.ops import rnn_cell
#modify by xjk
#from tensorflow.python.ops import rnn
from tensorflow.contrib import rnn
from tensorflow.python.ops import init_ops

__author__ = "Yuanpeng Zhang"
__date__   = "$2016-11-17$"


class tgGraph(object):
    def __init__(self
                 , hiddensize
                 , batchsize
                 , numsteps
                 , targetdelay
                 , projectsize
                 , maxgradnorm
                 , numlayers
                 , embeddingnumpy
                 , lr=0.0015
                 , momentum=0.85
                 , istrain=True
                 , distributed=False
                 , global_step=None):
        self.batchsize = batchsize
        self.numsteps = numsteps
        self.targetdelay = targetdelay

        size = hiddensize
        targetsize = len(embeddingnumpy)
        embeddingdim = len(embeddingnumpy[0])

        ## placeholder
        self._input_data = tf.placeholder(tf.int32, [batchsize, numsteps])
        self._targets = tf.placeholder(tf.int64, [numsteps, batchsize])
        self._lengths = tf.placeholder(tf.int32, [batchsize])
        self._frame_weights = tf.placeholder(tf.float32, [numsteps, batchsize])

        ## build cell
        forward  = rnn_cell.LSTMCell(size
                                     , num_proj=projectsize
                                     , use_peepholes=True
                                     , forget_bias=0.0)

        ## build wordembedding input id
        with tf.device("/cpu:0"):
            self.embedding = tf.get_variable("embedding", initializer=tf.constant(embeddingnumpy))
        inputs = tf.nn.embedding_lookup(self.embedding, self._input_data)

        #modify by xjk
        #embedding_table = tf.split(1, numsteps, inputs)
        embedding_table = tf.split(inputs, numsteps, 1)
        concat_embeddings = []
        for i in range(numsteps + targetdelay):
            #modify by xjk
            #concat_embeddings.append(tf.reshape(tf.concat(0, embedding_table[i])
                                                #, [batchsize, embeddingdim]))
            concat_embeddings.append(tf.reshape(tf.concat(embedding_table[i], 0)
                                                            , [batchsize, embeddingdim]))            

        ## build target
        self.targets = tf.reshape(self._targets, [batchsize * numsteps])
        self.onehot = tf.one_hot(self.targets, targetsize, 1.0, 0.0)
        #modify by xjk
        #self.lengths = tf.reshape(tf.concat(0, self._lengths), [batchsize])
        self.lengths = tf.reshape(tf.concat(self._lengths, 0), [batchsize])
        #modify by xjk
        #self.frame_weights = tf.reshape(tf.concat(0, self._frame_weights), [batchsize * numsteps])
        self.frame_weights = tf.reshape(tf.concat(self._frame_weights, 0), [batchsize * numsteps])
        #modify by xjk
        self.output = rnn.static_rnn(forward, concat_embeddings, dtype=tf.float32, sequence_length=self.lengths)

        ## softmax layer
        softmax_fw_w = tf.get_variable("softmax_fw_w", [projectsize, targetsize])
        softmax_fw_b = tf.get_variable("softmax_fw_b", [targetsize], initializer=init_ops.constant_initializer(0.0))

        #modify by xjk
        #logits_fw = tf.matmul(tf.concat(0, self.output[0]), softmax_fw_w) + softmax_fw_b
        logits_fw = tf.matmul(tf.concat(self.output[0], 0), softmax_fw_w) + softmax_fw_b

        ## build loss function, optimizer
        #modify by xjk
        #self.loss = tf.nn.softmax_cross_entropy_with_logits(logits_fw, self.onehot) * self.frame_weights
        self.loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits_fw, labels=self.onehot) * self.frame_weights
        self._cost = tf.reduce_sum(self.loss) / tf.reduce_sum(self.frame_weights)

        self._outputs = tf.argmax(logits_fw, 1)
        tvars = tf.trainable_variables()

        self.grads, _ = tf.clip_by_global_norm(tf.gradients(self._cost, tvars)
                                               , maxgradnorm)

        if istrain is True:
            optimizer = tf.train.GradientDescentOptimizer(lr)
            self._train_op = optimizer.apply_gradients(zip(self.grads, tvars))
            if distributed is True:
                tf.scalar_summary("cost", self._cost)
                self.summary_op = tf.merge_all_summaries()