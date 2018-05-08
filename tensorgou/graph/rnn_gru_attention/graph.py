"""
Tensorgou gru_att module graph define
"""

import tensorflow as tf

from tensorflow.python.ops import rnn_cell

__author__ = "Shuai Huo"
__date__   = "$2017-08-22$"


class tgGraph(object):
    def __init__(self
                 , hiddensize
                 , batchsize
                 , numsteps
                 , keep_prob
                 , numlayers
                 , numlabel
                 , max_grad_norm
                 , embeddingnumpy
                 , lr=0.0015
                 , momentum=0.85
                 , istrain=True
                 , distributed=False
                 , global_step=None):

        self.batchsize = batchsize
        self.numsteps = numsteps
        size = hiddensize
        #if lstm has num_proj,modify self.proj,otherwise self.proj=size
        self.proj=100
        self._input_data = tf.placeholder(tf.int32, [batchsize, numsteps])
        self._targets = tf.placeholder(tf.int32, [batchsize])
        #if reduce_max, use the second following line,otherwise the first
        self._lengths = tf.placeholder(tf.int32, [batchsize])
        #self._lengths=tf.placeholder(tf.float32,[numsteps,batchsize,100])

        # Slightly better results can be obtained with forget gate biases
        # initialized to 1 but the hyperparameters of the model would need to be
        # different than reported in the paper.
        #lstm_cell=rnn_cell.BasicLSTMCell(size)
        #lstm_cell = rnn_cell.LSTMCell(size, forget_bias=1.0,use_peepholes=True)
	gru_cell = rnn_cell.GRUCell(size)
        #lstm_cell = rnn_cell.LSTMCell(size, forget_bias=1.0,use_peepholes=True,num_proj=self.proj)
        #lstm_cell=tf.contrib.rnn.LayerNormBasicLSTMCell(size)
        #lstm_cell=tf.contrib.rnn.GridLSTMCell(size,use_peepholes=True)
        #lstm_cell=tf.contrib.rnn.NASCell(size)      
        #lstm_cell=tf.contrib.rnn.PhasedLSTMCell(size,use_peepholes=True)       
        #lstm_cell=tf.contrib.rnn.TimeFreqLSTMCell(size,use_peepholes=True,feature_size=100,frequency_skip=1)
        if istrain and keep_prob < 1:
            gru_cell = rnn_cell.DropoutWrapper(gru_cell, output_keep_prob=keep_prob)

        cell = rnn_cell.MultiRNNCell([gru_cell] * numlayers)

        self._initial_state = cell.zero_state(batchsize, tf.float32)
        
        #commit the following line if reduce_max
        offsets = (self._lengths - 1) * self.batchsize + tf.range(0, self.batchsize)

        with tf.device("/gpu:2"):
            #modify by xjk
            self.embedding = tf.get_variable("embedding", initializer=tf.constant(embeddingnumpy), trainable=False)
#            if(istrain):
#                self.embedding = tf.get_variable("embedding", initializer=tf.constant(embeddingnumpy))
#            else:
#                self.embedding = tf.get_variable("embedding")

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

        output = tf.reshape(tf.concat(outputs, 0), [-1, size])
        #add by xjk 
        # reduce_max method ; if not ,commit these 4 lines
        s=tf.split(output,num_or_size_splits=numsteps,axis=0)
        self.output2=s
        # if no need according to lengths , commit the following line
        #s=tf.multiply(s,self._lengths)
        #output=tf.reduce_max(s,0)       
        
        #commit the following line if reduce_max
        output = tf.gather(output, offsets)
        logits = tf.nn.xw_plus_b(output,
                                 tf.get_variable("softmax_w", [size, numlabel]),
                                 tf.get_variable("softmax_b", [numlabel]))
        self.results = tf.argmax(logits, 1)
        self.logits=logits
        batchsize = tf.size(self._targets)
        labels = tf.expand_dims(self._targets, 1)
        indices = tf.expand_dims(tf.range(0, batchsize), 1)
        concated = tf.concat([indices, labels],1 )
        onehot_labels = tf.sparse_to_dense(
            #modify by xjk tf.pack()->tf.stack
            concated, tf.stack([batchsize, numlabel]), 1.0, 0.0)
        #modify by xjk add "logits=,labels=" in paramer
        loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits,
                                                       labels=onehot_labels,
                                                       name='xentropy')

        corrects = tf.nn.in_top_k(logits, self._targets, 1)
        self._corrects_num = tf.reduce_sum(tf.cast(corrects, tf.int32))
        self._cost = cost = tf.reduce_mean(loss, name='xentropy_mean')
        self._final_state = states[-1]
    
        if istrain:
            self._lr = tf.Variable(0.0, trainable=False)
            tvars = tf.trainable_variables()
            grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars)
                                              , max_grad_norm)
            #change 1.0 to lr is no effect
            #optimizer = tf.train.GradientDescentOptimizer(1.0)  # TODO: No training effect if self.lr.
            optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
            self._train_op = optimizer.apply_gradients(zip(grads, tvars), global_step=global_step)
            if distributed is True:
                tf.scalar_summary("cost", self._cost)
                tf.scalar_summary("accuracy", self._corrects_num)
                self.summary_op = tf.merge_all_summaries()

