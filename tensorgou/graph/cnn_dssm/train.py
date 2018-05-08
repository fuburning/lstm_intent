"""
Define parameter for cnn_dssm
"""
import tensorflow as tf
import tensorgou.graph.cnn_dssm.graph as graph
import numpy as np
import tensorgou.logging as log

__author__ = "Yuanpeng Zhang"
__date__   = "$2016-10-18$"


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
    config.add_argument('modelsize', list, required=False
                        , default=[786, 512, 512, 256])
    config.add_argument('numsteps', int, required=False, default=30)
    config.add_argument('buildwordembedding', bool, required=False, default=True)


def validcheck(config_dict):
    if not config_dict.has_key("name"):
        raise Exception("No name be defined!")

    if config_dict.has_key("numbatchbuf"):
        numbatchbuf = config_dict["numbatchbuf"]
        if numbatchbuf < 100:
            raise Exception("Expect numbatchbuf >= 100, get {}".format(numbatchbuf))


def init(args):
    pass


def preloop(graph, sess):
    pass


def getGraph(embeddingnumpy, arguments, distributed=False, global_step=None):
    return graph.tgGraph(modelsize = arguments.modelsize
                         , batchsize = arguments.batchsize
                         , numsteps = arguments.numsteps
                         , embeddingnumpy = embeddingnumpy
                         , embeddingsize = arguments.embeddingsize
                         , lr = arguments.lr
                         , momentum = arguments.momentum
                         , istrain = True
                         , distributed=distributed
                         , global_step=global_step)


def sentence2id(sentences, word2id, numsteps):
    rows = len(sentences)
    cols = numsteps

    ids = np.zeros((rows, cols), dtype = np.int32)
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
        for icol in range(numsteps):
            ids[irow, icol] = wordids[icol % valid_words]

    return ids


def prenxtbatch(args, graph, dataset, word2id, global_step = None):
    querys, title_a, title_b = dataset.get_batch()
    querysid = sentence2id(querys, word2id, args.numsteps)
    ta_ids = sentence2id(title_a, word2id, args.numsteps)
    tb_ids = sentence2id(title_b, word2id, args.numsteps)
    minibatch = np.concatenate((querysid, ta_ids, tb_ids), axis = 0)

    input_id2 = minibatch
    input_id1 = np.roll(input_id2, 1, axis = 1)
    input_id0 = np.roll(input_id1, 1, axis = 1)
    if args.distributed is False:
        fetches = [graph._cost, graph._corrects_num, graph.train_op_]
    else:
        fetches = [graph._cost,
                   graph._corrects_num,
                   graph.train_op_,
                   graph.summary_op,
                   global_step]
    feed_dict = {graph.inputdata[0] : input_id0, graph.inputdata[1] : input_id1, graph.inputdata[2] : input_id2}

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





