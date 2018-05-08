"""
Tensorgou Train script
"""

from tensorgou.io.wordembedding import getWordembedding
from tensorgou.logging import Logging
from tensorgou.configure.configuration import setParameter
import tensorgou.utils.namekey as nk

import time
import os
import tensorflow as tf

__author__ = "Yuanpeng Zhang"
__date__   = "$2016-11-03$"

log = Logging()


def initDistributed(args):
    log.message("Init distributed cluster ......")
    ps_hosts = args.pshosts.split(",")
    worker_hosts = args.workerhosts.split(",")
    numworkers = len(worker_hosts)
    setParameter(args, "numworker", numworkers, int, True)

    cluster = tf.train.ClusterSpec({"ps": ps_hosts, "worker" : worker_hosts})
    dict = cluster.as_dict()
    log.message("Total {} worker join ......".format(numworkers))
    log.message("Cluster Dict: ")
    log.message("{}".format(dict))

    log.message("Create server ......")

    ## create server
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True

    server = tf.train.Server(cluster
                             , job_name = args.job_name
                             , task_index = args.task_index
                             , config = config)

    return server, cluster


def initworker(args, server, cluster, inmodule):
    ## Step 1: build wordembedding
    log.message("Build wordembedding ......")
    word2id, embedding = getWordembedding(args)

    ## build graph
    inmodule.init(args)
    with tf.Graph().as_default(), tf.device(tf.train.replica_device_setter(
            worker_device = "/job:worker/task:%d" % args.task_index
            , cluster = cluster)):
        ## Init global_step
        with tf.device('/cpu:0'):
            global_step = tf.get_variable('global_step', [],
                                          initializer = tf.constant_initializer(0), trainable = False)

        ## Init model
        with tf.device("/gpu:{}".format(args.deviceid)):
            initializer = tf.random_uniform_initializer(-1.0 * args.randomrange, args.randomrange)
            with tf.variable_scope("model", reuse=None, initializer=initializer):
                graph = inmodule.getGraph(embedding, args
                                          , distributed=args.distributed
                                          , global_step=global_step)

        saver = tf.train.Saver(max_to_keep=args.max_to_keep
                               , keep_checkpoint_every_n_hours=args.keep_checkpoint_every_n_hours)
        init_op = tf.initialize_all_variables()

        ## create supervisor
        log.message("Create supervisor ......")
        sv = tf.train.Supervisor(is_chief = args.ischief,
                                 logdir = args.output,
                                 saver = saver,
                                 summary_op=graph.summary_op,
                                 checkpoint_basename=nk.sKeyckptfnm,
                                 init_op = init_op,
                                 save_model_secs=0,
                                 global_step = global_step)

        ## create session
        log.message("Build session ......")
        configproto = tf.ConfigProto()
        configproto.gpu_options.allow_growth = True
        configproto.log_device_placement=args.log_device
        configproto.allow_soft_placement=args.soft_placement
        configproto.inter_op_parallelism_threads=args.num_cores
        configproto.intra_op_parallelism_threads=args.num_cores

        if args.ischief is True:
            log.message("Worker {}: Initializing session...".format(args.task_index))
        else:
            log.message("Worker {}: Waiting for session to be initialized...".format(args.task_index))

        sess = sv.prepare_or_wait_for_session(server.target
                                              , config=configproto
                                              , start_standard_services=True)

        return word2id, graph, sess, saver, global_step


def trainLoop(args, inmodule, graph, sess, saver, dataset, word2id, global_step):
    evalSystem = inmodule.evalSystem(args.batchsize)
    inmodule.preloop(graph, sess)

    writer = tf.train.SummaryWriter(args.output, graph=tf.get_default_graph())

    log.message("Start train loop ......")
    log.message("----------------------------")
    results = []
    loops = 0

    gstartT = time.time()
    while 1:
        ## Step 1: get nxt dataset
        try:
            fetches, feed_dict = inmodule.prenxtbatch(args=args
                                                      , graph=graph
                                                      , dataset=dataset
                                                      , word2id=word2id
                                                      , global_step=global_step)

            result = sess.run(fetches, feed_dict)
            inmodule.postProcess(result)
            ##TODO: move writerSummer into postProcess
            evalSystem.writerSummer(writer, result)

            results.append(result)
            loops += 1

            if loops % 10 == 1:
                stop = evalSystem.evalit(loops, results)
                results[:] = []
                if stop is True:
                    dataset.setstop()

            if args.ischief and (args.ckptperbatch > 0) and (loops % args.ckptperbatch == 0):
                fnm = str("{}_{}".format(nk.sKeyckptfnm, loops))
                ckptfnm = os.path.join(args.output, fnm)
                log.message("Saving checkpoint model: {} ......".format(ckptfnm))
                saver.save(sess, ckptfnm)

        except StopIteration:
            break

    gstopT = time.time()
    lenT = gstopT - gstartT
    log.message("Run Total Time: {:.2}s".format(lenT))
    if loops != 0:
        log.message("Speed per batch: {:.2}s".format(lenT / loops))
