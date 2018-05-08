"""
Define parameter for rnn_lstm_query
"""
import tensorflow as tf
import tensorgou.graph.rnn_lstm_query.graph as graph
import numpy as np
import tensorgou.logging as log
from tensorgou.configure.configuration import setParameter

__author__ = "Yuanpeng Zhang"
__date__   = "$2016-11-17$"

"""
Define global variable which will be used special method
"""
class g_variable(object):
    g_targets = None
    g_lengths = None
    g_frameweight = None
    g_ncorrect = 0

G_Vals = g_variable()


class Config(object):
    """Large config."""
    max_grad_norm = 5
    num_layers = 2
    num_steps = 5
    target_delay = 0
    hidden_size = 256
    project_size = 128
    keep_prob = 0.35
    lr_decay = 0.9


def get_model(args):
    return Config()


def getconfigure(config):
    config.add_argument('trainfnms', str, required=True
                        , helps="Training data file list")
    config.add_argument('testfnms', str, required=False, default=None
                        , helps="Test data file list")
    config.add_argument('wordlistfnm', str, required=False, default=None
                        , helps="Word list file of word_embedding")
    config.add_argument('checkwordlist', bool, required=False, default=False
                        , helps="Scan wordlist, make sure the word is unique")
    config.add_argument('worddictfnm', str, required=False, default=None
                        , helps="Word dict file 'word2vec'")
    config.add_argument('embeddingsize', int, required=False, default=100)
    config.add_argument('buildwordembedding', bool, required=False, default=True)


def validcheck(config_dict):
    pass


def init(args):
    model = get_model(args)
    setParameter(args, "maxgradnorm", model.max_grad_norm, int, True)
    setParameter(args, "numlayers", model.num_layers, int, True)
    setParameter(args, "numsteps", model.num_steps, int, True)
    setParameter(args, "targetdelay", model.target_delay, int, True)
    setParameter(args, "hiddensize", model.hidden_size, int, True)
    setParameter(args, "projectsize", model.project_size, int, True)
    setParameter(args, "keepprob", model.keep_prob, float, True)
    setParameter(args, "lrdecay", model.lr_decay, float, True)


def preloop(graph, sess):
    pass


def getGraph(embeddingnumpy, arguments, distributed=False, global_step=None):
    return graph.tgGraph(hiddensize=arguments.hiddensize
                         , batchsize=arguments.batchsize
                         , numsteps=arguments.numsteps
                         , targetdelay=arguments.targetdelay
                         , projectsize=arguments.projectsize
                         , maxgradnorm=arguments.maxgradnorm
                         , numlayers=arguments.numlayers
                         , embeddingnumpy=embeddingnumpy
                         , lr = arguments.lr
                         , momentum=arguments.momentum
                         , istrain = True
                         , distributed=distributed
                         , global_step=global_step)


def sentence2id(sentence, word2id):
    words = sentence.split(' ')

    cols = len(words)
    if cols == 0:
        raise Exception("Bad data! Empty line in train file?")

    wordids = []
    wildid = word2id['</s>']
    for word in words:
        if word in word2id:
            wordids.append(word2id[word])
        else:
            wordids.append(wildid)

    return wordids


def preprocess(querys, answers, word2id, batchsize, numsteps):
    print numsteps
    print numsteps
    sents = np.zeros([batchsize, numsteps], dtype = np.int32)
    targets = np.zeros([numsteps, batchsize], dtype = np.int64)
    lengths = np.zeros([batchsize], dtype = np.int32)
    frameweight = np.zeros([numsteps, batchsize], dtype = np.float32)

    assert len(querys) == batchsize
    assert len(answers) == batchsize

    ## process pairs
    for iline in range(batchsize):
        ## convert to id
        q_wids = sentence2id(querys[iline], word2id)
        r_wids = sentence2id(answers[iline], word2id)

        q_length = len(q_wids)
        r_length = len(r_wids)
        ## clip size
        qlen = numsteps - q_length - 1
        if qlen > 0:
            rlen = min(qlen, r_length)
        else:
            rlen = 0

        if not qlen > 0:
            q_wids = q_wids[:numsteps - 1]
            r_wids = []

        if rlen > 1:
            r_wids = r_wids[:rlen - 1]
        else:
            r_wids = []
        q_length = len(q_wids)
        r_length = len(r_wids)
        total_len = q_length + r_length + 2

        ## push query && answer to sents
        for i in range(q_length):
            sents[iline, q_length - i - 1] = q_wids[i]
        cursor = q_length
        sents[iline, cursor] = 1

        if r_length > 0:
            cursor += 1
            for i in range(r_length):
                sents[iline, cursor + i] = r_wids[i]
            cursor += r_length
            sents[iline, cursor] = 2

        ## push query && answer to target
        cursor = 0
        for i in range(q_length - 1):
            targets[i, iline] = q_wids[i + 1]
        cursor = q_length - 1
        targets[cursor, iline] = 1

        if r_length > 0:
            cursor += 1
            for i in range(r_length):
                targets[cursor + i, iline] = r_wids[i]
            cursor += r_length
            targets[cursor, iline] = 2

        ## build length && frame info
        lengths[iline] = total_len
        for i in range(q_length, total_len - 1):
            frameweight[i, iline] = 1.0

    return sents, targets, lengths, frameweight


def prenxtbatch(args, graph, dataset, word2id, global_step = None):
    querys, answers = dataset.get_batch()
    print querys
    print answers
    q=[]
    for i in range(len(querys)):
        print querys[i],
        print " ",
    print()
    
    if not len(querys) == len(answers):
        raise Exception("Input data error, cant match query 2 answer")
    if len(querys) != args.batchsize:
        raise Exception("Bad input data size {} != batch size {}".format(len(querys), args.batchsize))

    sents, G_Vals.g_targets, G_Vals.g_lengths, G_Vals.g_frameweight = preprocess(querys
                                                                                 , answers
                                                                                 , word2id
                                                                                 , args.batchsize
                                                                                 , args.numsteps)
    print sents
    print G_Vals.g_targets
    if args.distributed is False:
        fetches = [graph._cost, graph._outputs, graph._train_op]
    else:
        fetches = [graph._cost,
                   graph._outputs,
                   graph._train_op,
                   graph.summary_op,
                   global_step]
    feed_dict = {graph._input_data : sents
                 , graph._targets : G_Vals.g_targets
                 , graph._lengths : G_Vals.g_lengths
                 , graph._frame_weights : G_Vals.g_frameweight}

    return fetches, feed_dict


def postProcess(result):
    pass


class evalSystem(object):
    def __init__(self, batchsize):
        self.steps = 0
        self.cost = 0.0
        self.correctnum = 0
        self.stop = False
        self.batchsize = batchsize
        if self.batchsize <= 0:
            raise Exception("Bad batchsize {}? Expect > 0".format(batchsize))

    def writerSummer(self, writer, result):
        assert len(result) == 5
        writer.add_summary(result[3], result[4])

    def evalit(self, loops, results):
        l_cost = 0.0
        l_correctnum = 0
        l_totalnum = 0
        l_step = loops - self.steps
        self.steps = loops

        for item in results:
            if len(item) < 2:
                raise Exception("Bad result type! Expect {}, get {}"
                                .format(3, len(item)))
            self.cost += item[0]

        ## cal correct
        nr = len(results)
        #add by xjk
        out=open("/search/odin/tensorflow/lstm_output/out",'a')
        if nr > 0:
            numsteps = Config().num_steps
            output = results[nr - 1][1]
            l_cost = results[nr - 1][0]
            for j in range(self.batchsize):
                for k in range(numsteps):
                    if k < G_Vals.g_lengths[j] and G_Vals.g_frameweight[k, j] > 0:
                        if G_Vals.g_targets[k, j] == output[k * self.batchsize + j]:
                            # add by xjk
                            out.write(str(G_Vals.g_targets[k, j])+" "+str(output[k * self.batchsize + j])+'\n')
                            l_correctnum += 1

                        l_totalnum += 1

        l_accuracy = float(l_correctnum) / float(l_totalnum)

        count = self.steps * self.batchsize
        if count <= 0:
            raise Exception("Bad l_step {}? Expect > 0".format(l_step))

        l_avgcost = self.cost / float(count)

        log.message("No.{} batch: curr_lost[{:.5f}]    curr_prec[{:.6f}]    ave_loss[{:.6f}]"
                    .format(loops + 1, l_cost, l_accuracy, l_avgcost))

        return False
