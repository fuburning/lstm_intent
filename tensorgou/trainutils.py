"""
Tensorgou Train script
"""

from tensorgou.io.wordembedding import getWordembedding
from tensorgou.logging import Logging
import tensorgou.utils.namekey as nk

import time
import tensorflow as tf
import os
import pickle

__author__ = "Yuanpeng Zhang"
__date__   = "$2016-10-19$"

log = Logging()


def initenviroment(args):
    ## if ps set device to none
    if args.__contains__("isps"):
        if args.isps is True:
            os.environ["CUDA_VISIBLE_DEVICES"] = " "
            return

    gpuid = int(args.deviceid)

    if (gpuid < 0) or (gpuid > 7):
        raise Exception("Bad deviceid value {}, expect 0-7".format(gpuid))

    log.message("Setup gpu id: {}".format(gpuid))
    os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(gpuid)


def inittensorflow(args, inmodule):
    ## Step 1: build wordembedding
    log.message("Build wordembedding ......")
    word2id, embedding = getWordembedding(args)
    # add by xjk
    #emb_file=open("/search/huoshuai/xiaojiakun/tensorflow/embedding",'w')
    #pickle.dump(embedding,emb_file)
    ## Step 2: build graph
    log.message("Build graph ......")
    inmodule.init(args)
    with tf.device("/gpu:{}".format(args.deviceid)):
        initializer = tf.random_uniform_initializer(-1.0 * args.randomrange, args.randomrange)
        with tf.variable_scope("model", reuse=None, initializer=initializer):
            graph = inmodule.getGraph(embedding, args)
            #add by xjk
        with tf.variable_scope("model", reuse=True, initializer=initializer):
            graph_test=inmodule.getGraph(embedding,args,istrain=False)

    ## step3: create sess
    log.message("Build session ......")
    configproto = tf.ConfigProto()
    configproto.gpu_options.allow_growth = True
    configproto.log_device_placement=args.log_device
    configproto.allow_soft_placement=args.soft_placement
    configproto.inter_op_parallelism_threads=args.num_cores
    configproto.intra_op_parallelism_threads=args.num_cores

    sess = tf.Session(config = configproto)

    ## Step 4: Saver
    saver = tf.train.Saver(write_version=tf.train.SaverDef.V1,max_to_keep=args.max_to_keep
                           , keep_checkpoint_every_n_hours=args.keep_checkpoint_every_n_hours)

    ## Step 5: Init variable
    if args.checkpoint is not None:
        if not os.path.exists(args.checkpoint):
            raise Exception("Can't find checkpoint file '{}'?".format(args.checkpoint))
        log.message("Restore from checkpoint '{}' ......".format(args.checkpoint))
        saver.restore(sess, args.checkpoint)
    else:
        log.message("Init variables ......")
        init = tf.initialize_all_variables()
        sess.run(init)
        #log.message("Setup word embedding ......")
        #sess.run(graph.embedding.assign(graph.embeddingnumpy))
    #add "graph_test" by xjk
    return word2id, graph, sess, saver,graph_test

def trainLoop(args, inmodule, graph, sess, saver, dataset, word2id):
    evalSystem = inmodule.evalSystem(args.batchsize)
    inmodule.preloop(graph, sess)

    log.message("Start train loop ......")
    log.message("----------------------------")
    results = []
    loops = 0

    gstartT = time.time()
    while 1:

        ## Step 1: get nxt dataset
        #add try by xjk
        try:
            fetches, feed_dict = inmodule.prenxtbatch(args=args
                                                  , graph=graph
                                                  , dataset=dataset
                                                  , word2id=word2id)
        except Exception,e:
            break
        result = sess.run(fetches, feed_dict)
        inmodule.postProcess(result)

        results.append(result)
        loops += 1
        #print("loops:"+"  "+str(loops))
        if loops % 10 == 1:
            stop = evalSystem.evalit(loops, results)
            results[:] = []
            if stop is True:
                dataset.setstop()
                
        if (args.ckptperbatch > 0) and (loops % args.ckptperbatch == 0):
            fnm = str("{}_{}".format(nk.sKeyckptfnm, loops))
            ckptfnm = os.path.join(args.output, fnm)
            log.message("Saving checkpoint model: {} ......".format(ckptfnm))
            saver.save(sess, ckptfnm)


    gstopT = time.time()
    lenT = gstopT - gstartT
    log.message("Run Total Time: {:.2}s".format(lenT))
    if loops != 0:
        log.message("Speed per batch: {:.2}s".format(lenT / loops))
#add by xjk
def testLoop(args, inmodule, graph, sess, saver, word2id):
    evalSystem = inmodule.evalSystem(args.batchsize)
    inmodule.preloop(graph, sess)

    log.message("Start test loop ......")
    log.message("----------------------------")
    results = []
    results_write=[]
    loops = 0

    gstartT = time.time()
    ## Step 1: get nxt dataset
    fetches, feed_dict, input_data = inmodule.pretestbatch(args=args
                                                  , graph=graph
                                                  , word2id=word2id)
    #remove the last feed_dict
    
    #input_data_file=open("/search/huoshuai/xiaojiakun/tensorflow/input_data",'w')
    #pickle.dump(input_data[0],input_data_file)    
    #input_data_file.close()
    for i in range(0,len(feed_dict)-1):
        result = sess.run(fetches, feed_dict[i])
        #inmodule.postProcess(result)

        results.append(result)
        results_write.append(result)
        loops += 1
        #print("loops:"+"  "+str(loops))
        if loops % 10 == 1:
            stop = evalSystem.evalit(loops, results,istrain=False)
            results[:] = []
            if stop is True:
                print"stop is True!"
                break
    out=open("/search/huoshuai/recency_model/out_valid",'w')
    for i in range(len(results_write)):
        for j in range(len(results_write[i][0])):
            #out.write(str(results_write[i][0][j])+'\n')
            out.write(str(results_write[i][0][j][0])+'\t'+str(results_write[i][0][j][1])+'\n')
   # with open("/search/huoshuai/xiaojiakun/tensorflow/out_valid",'w') as out:
       # for line in results:
            #out.write(line)

    gstopT = time.time()
    lenT = gstopT - gstartT
    log.message("Run Total Time: {:.2}s".format(lenT))
    if loops != 0:
        log.message("Speed per batch: {:.2}s".format(lenT / loops))







