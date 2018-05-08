"""
Define parameter for rnn_relevance
"""
import tensorflow as tf
import tensorgou.graph.rnn_relevance.graph as graph
import numpy as np
import tensorgou.logging as log
from tensorgou.configure.configuration import setParameter

__author__ = "Yuanpeng Zhang"
__date__   = "$2016-11-08$"


class SmallConfig(object):
    """Small config."""
    max_grad_norm = 5
    num_layers = 2
    num_steps = 30
    hidden_size = 200
    max_epoch = 4  # Frequency to decay learning rate.
    keep_prob = 1.0
    lr_decay = 0.5


class MediumConfig(object):
    """Medium config."""
    max_grad_norm = 5
    num_layers = 2
    num_steps = 35
    hidden_size = 650
    max_epoch = 6
    keep_prob = 0.5
    lr_decay = 0.8


class LargeConfig(object):
    """Large config."""
    max_grad_norm = 10
    num_layers = 2
    num_steps = 35
    hidden_size = 1500
    max_epoch = 14
    keep_prob = 0.35
    lr_decay = 1 / 1.15


class TestConfig(object):
    """Tiny config, for testing."""
    init_scale = 0.1
    learning_rate = 1.0
    max_grad_norm = 1
    num_layers = 1
    num_steps = 2
    hidden_size = 2
    max_epoch = 1
    max_max_epoch = 1
    keep_prob = 1.0
    lr_decay = 0.5
    batch_size = 20
    vocab_size = 10000


def get_model(args):
    if args.model == "small":
        return SmallConfig()
    elif args.model == "medium":
        return MediumConfig()
    elif args.model == "large":
        return LargeConfig()
    elif args.model == "test":
        return TestConfig()
    else:
        raise ValueError("Invalid model: %s", args.model)


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
    config.add_argument('model', str, required=False, default="small"
                        , helps="Supported model configurations: small, medium, large")
    config.add_argument('buildwordembedding', bool, required=False, default=True)


def validcheck(config_dict):
    pass


def init(args):
    model = get_model(args)
    setParameter(args, "numsteps", model.num_steps, int, True)
    setParameter(args, "hiddensize", model.hidden_size, int, True)
    setParameter(args, "maxgradnorm", model.max_grad_norm, int, True)
    setParameter(args, "keepprob", model.keep_prob, float, True)
    setParameter(args, "numlayers", model.num_layers, int, True)
    setParameter(args, "lrdecay", model.lr_decay, float, True)
    setParameter(args, "lrepoch", model.max_epoch, int, True)


def preloop(graph, sess):
    pass


def getGraph(embeddingnumpy, arguments, distributed=False, global_step=None):
    return graph.tgGraph(hiddensize=arguments.hiddensize
                         , batchsize=arguments.batchsize
                         , numsteps=arguments.numsteps
                         , keep_prob=arguments.keepprob
                         , numlayers=arguments.numlayers
                         , embeddingnumpy=embeddingnumpy
                         , lr = arguments.lr
                         , momentum=arguments.momentum
                         , istrain = True
                         , distributed=distributed
                         , global_step=global_step)


def sentence2id(sentences, word2id, numsteps):
    rows = len(sentences)
    cols = numsteps

    ids = np.zeros((rows, cols), dtype = np.int32)
    length = np.zeros(rows, dtype = np.int32)
    wildid = word2id['</s>']
    for irow in range(rows):
        words = sentences[irow].split(' ')
        wordids = []
        for word in words:
            if word in word2id:
                wordids.append(word2id[word])
            else:
                wordids.append(wildid)
        valid_words = len(wordids)
        length[irow] = valid_words
        for icol in range(numsteps):
            ids[irow, icol] = wordids[icol % valid_words]

    return ids, length


def prenxtbatch(args, graph, dataset, word2id, global_step = None):
    querys, title_a, title_b = dataset.get_batch()
    querysid, qlength = sentence2id(querys, word2id, args.numsteps)
    ta_ids, talength = sentence2id(title_a, word2id, args.numsteps)
    tb_ids, tblength = sentence2id(title_b, word2id, args.numsteps)
    ids = np.concatenate((querysid, ta_ids, tb_ids), axis = 0)
    idslen = np.concatenate((qlength, talength, tblength), axis = 0)

    if args.distributed is False:
        fetches = [graph._cost, graph._corrects_num, graph.train_op_]
    else:
        fetches = [graph._cost,
                   graph._corrects_num,
                   graph.train_op_,
                   graph.summary_op,
                   global_step]
    feed_dict = {graph._input_data : ids, graph._lengths : idslen}

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
        l_step = loops - self.steps
        self.steps = loops

        for item in results:
            if len(item) < 2:
                raise Exception("Bad result type! Expect {}, get {}"
                                .format(3, len(item)))
            l_cost += item[0]
            l_correctnum += item[1]

        count = l_step * self.batchsize
        if count <= 0:
            raise Exception("Bad l_step {}? Expect > 0".format(l_step))

        l_avgcost = l_cost / count
        l_accuracy = l_correctnum / count

        log.message("No.{} batch: Cost[{:.5f}]    Accuracy[{:.5f}]"
                    .format(loops + 1, l_avgcost, l_accuracy))

        return False