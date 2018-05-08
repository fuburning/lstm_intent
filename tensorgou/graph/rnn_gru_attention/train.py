"""
Define parameter for rnn_relevance
"""
import tensorflow as tf
import tensorgou.graph.rnn_classification.graph as graph
import numpy as np
import tensorgou.logging as log
from tensorgou.configure.configuration import setParameter

__author__ = "Yuanpeng Zhang"
__date__   = "$2016-11-08$"


class g_variable(object):
    state = None

G_Vals = g_variable()


class SmallConfig(object):
    """Small config."""
    max_grad_norm = 5
    num_layers = 1
    num_steps = 8
    hidden_size = 100
    max_epoch = 1  # Frequency to decay learning rate.
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
    batch_size = 100
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
    config.add_argument('numlabel', int, required=False, default=2
                        , helps="number of label for classification")


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
    G_Vals.state = sess.run(graph._initial_state)

#add istrain=True in paramer by xjk
def getGraph(embeddingnumpy, arguments, distributed=False, global_step=None,istrain=True):
    return graph.tgGraph(hiddensize=arguments.hiddensize
                         , batchsize=arguments.batchsize
                         , numsteps=arguments.numsteps
                         , keep_prob=arguments.keepprob
                         , numlayers=arguments.numlayers
                         , numlabel=arguments.numlabel
                         , max_grad_norm=arguments.maxgradnorm
                         , embeddingnumpy=embeddingnumpy
                         , lr = arguments.lr
                         , momentum=arguments.momentum
                         , istrain = istrain
                         , distributed=distributed
                         , global_step=global_step)


def sentence2id(sentences, labellist, word2id, numsteps):
    llist = []
    idslist = []
    wildid = word2id['</s>']
    for s in sentences:
        words = s.split(' ')
        if len(words) < 1:
            raise Exception("Bad Input data, empty?")
        wordids = []
        for word in words:
            if word in word2id:
                wordids.append(word2id[word])
            else:
                wordids.append(wildid)
        idslist.append(wordids)

    # over-length sentence cut here
    numsentence = len(idslist)
    data = np.zeros([numsentence, numsteps], dtype=np.int32)
    length = np.zeros([numsentence], dtype=np.int32)
    for i in range(numsentence):
        arr = np.asarray(idslist[i])
        length[i] = min(numsteps, len(idslist[i]))
        arr.resize(numsteps)
        data[i] = arr

    for l in labellist:
        tags = l.split(' ')
        if len(tags) > 1:
            print tags
            raise Exception("Support only one label in current version! Found {}"
                            .format(len(tags)))
        llist.append(tags[0])

    return data, length, llist


def prenxtbatch(args, graph, dataset, word2id, global_step = None):
    sentencelist, labellist = dataset.get_batch()
    ids, idlen, labels = sentence2id(sentencelist, labellist, word2id, args.numsteps)
    
    #build length mat
    pos=[1 for i in range(100)]
    neg=[0 for i in range(100)]
    length_tensor=[[]for i in range(args.numsteps)]
    for i in range(args.batchsize):
        temp=idlen[i]
        
        #commit the following three line if use reduce_max ,otherwise reduce_average
        #pos_val=1/float(temp)
        #pos_val=round(pos_val,4)
        #pos=[pos_val for i in range(100)]    
        
        for j in range(args.numsteps):
            if(temp>0):
                length_tensor[j].append(pos)
            else:
                length_tensor[j].append(neg)
            temp-=1    
            
    if args.distributed is False:
        fetches = [graph.results,
                   graph._cost,
                   graph._corrects_num,
                   graph._train_op,
                   graph._final_state]
    else:
        fetches = [graph.results,
                   graph._cost,
                   graph._corrects_num,
                   graph._train_op,
                   graph._final_state,
                   graph.summary_op,
                   global_step]
    feed_dict = {graph._input_data : ids
                 , graph._targets : labels
                 #change graph._lengths if output format varies(length_tensor or idlen)
                 , graph._lengths : idlen
                 , graph._initial_state : G_Vals.state}

    return fetches, feed_dict

def pretestbatch(args, graph, word2id, global_step = None):
    sentencelist, labellist = get_test(args.batchsize)
    if args.distributed is False:
        fetches = [graph.logits,
                       graph._cost,
                       graph._corrects_num,
                       graph._final_state]
    else:
        fetches = [graph.results,
                       graph._cost,
                       graph._corrects_num,
                       graph._final_state,
                       global_step]    
    sentence_len=len(sentencelist)
    
    ids_list=[];idlen_list=[];labels_list=[];
    feed_dict=[]
    input_data=[]
    pos=[1 for i in range(100)]
    neg=[0 for i in range(100)]
      
    for i in range(sentence_len-1):
        ids, idlen, labels = sentence2id(sentencelist[i], labellist[i], word2id, args.numsteps)

        #build length mat
        length_tensor=[[]for i in range(args.numsteps)]  
        for i in range(args.batchsize):
            temp=idlen[i]
            
            #commit the following three line if use reduce_max ,otherwise reduce_average
            #pos_val=1/float(temp)
            #pos_val=round(pos_val,4)
            #pos=[pos_val for i in range(100)]        
            
            for j in range(args.numsteps):
                if(temp>0):
                    length_tensor[j].append(pos)
                else:
                    length_tensor[j].append(neg)
                temp-=1            
                
        feed_dict.append({graph._input_data : ids
                 , graph._targets : labels
                #change graph._lengths if output format varies(length_tensor or idlen)
                 , graph._lengths : idlen
                 , graph._initial_state : G_Vals.state})
        input_data.append([ids,labels,idlen,G_Vals.state])

    return fetches, feed_dict,input_data
#add by xjk
def get_test(batchsize):
    sentence=[[]]
    label=[[]]
    batch_num=0
    count=0
    with open("/search/huoshuai/recency_model/valid") as t:
        for line in t:
            count+=1
            sentence[batch_num].append(line.split('\n')[0].split('\t')[0])
            label[batch_num].append(line.split('\n')[0].split('\t')[1])
            if(count>=batchsize):
                count=0
                sentence.append([])
                label.append([])
                batch_num+=1
    return sentence,label

def postProcess(result):
    if not len(result) >= 5:
        raise Exception("Bad result type! Expect >= 5, get {}".format(len(result)))

    G_Vals.state = result[4]


class evalSystem(object):
    def __init__(self, batchsize):
        self.steps = 0
        self.correctnum = 0
        self.stop = False
        self.batchsize = batchsize
        if self.batchsize <= 0:
            raise Exception("Bad batchsize {}? Expect > 0".format(batchsize))

    def writerSummer(self, writer, result):
        assert len(result) == 7
        writer.add_summary(result[5], result[6])
     #add istrain=true by xjk  
    def evalit(self, loops, results,istrain=True):
        l_correctnum = 0
        l_step = loops - self.steps
        l_cost = 0.0
        self.steps = loops

        for item in results:
            #add istrain by xjk
            if(istrain):
                if len(item) < 5:
                    raise Exception("Bad result type! Expect >= 5, get {}"
                                .format(len(item)))
            self.correctnum += item[2]
            l_correctnum += item[2]
            l_cost += item[1]
        if len(results) > 0:
            l_cost /= float(len(results))

        t_count = self.steps * self.batchsize
        l_count = l_step * self.batchsize
        if l_count <= 0:
            raise Exception("Bad l_step {}? Expect > 0".format(l_step))

        l_accuracy = (l_correctnum * 1.0) / float(l_count)
        t_accuracy = (self.correctnum * 1.0) / float(t_count)

        log.message("No.{} batch: local cost [{:.5f}]    total accuracy[{:.5f}]    local accuracy[{:.5f}]"
                    .format(loops + 1, l_cost, t_accuracy, l_accuracy))

        return False
