"""
Tensorgou Train script
"""
import sys
import os
import argparse

from tensorgou.configure.configuration import setParameter
from tensorgou.logging import Logging
import tensorgou.configure.configuration as configuration
from tensorgou.trainutils import initenviroment
from tensorgou.dtrainutils import initworker, trainLoop, initDistributed
from tensorgou.io.datasets import initDataset

__author__ = "Yuanpeng Zhang"
__date__   = "$2016-11-01$"


def parseArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_fnm', type=str
                        , help="INI type configure file which should has 'main' section")
    parser.add_argument('--job_name', type=str, default="worker"
                        , help="One of 'ps', 'worker'")
    parser.add_argument('--task_index', type=int, default=0
                        , help="Index of task within the job")
    parser.add_argument('--deviceid', type=int, default=-1
                        , help="GPU ID: 0-7")
    args = parser.parse_args()
    config, inmodule = create_config(args.config_fnm)

    ## add cmd args
    setParameter(config, 'deviceid', args.deviceid, int, True)
    setParameter(config, 'task_index', args.task_index, int, True)
    setParameter(config, 'job_name', args.job_name, str, True)
    if args.task_index == 0:
        setParameter(config, 'ischief', True, bool, True)
    else:
        setParameter(config, 'ischief', False, bool, True)

    if args.job_name == "ps":
        setParameter(config, 'isps', True, bool, True)
    else:
        setParameter(config, 'isps', False, bool, True)

    return config, inmodule


def create_config(config_file):
    config = configuration.Configuration()
    config.add_argument('name', str, required=True
                        , helps="graph name: cnn_dssm, rnn ...")
    config.add_argument('type', str, required=False, default="train")
    config.add_argument('distributed', bool, required=False, default=True)
    config.add_argument('workerhosts', str, required=True
                        , helps="Comma-separated list of hostname:port pairs")
    config.add_argument('pshosts', str, required=True
                        , helps="Comma-separated list of hostname:port pairs")
    config.add_argument('output', str, required=True
                        , helps="Result save path")
    config.add_argument("num_cores", int, required=False, default=1
                        , helps="Number of CPU cores to be used. Default: All")
    config.add_argument("log_device", bool, required=False, default=False
                        , helps="Log device placement or not")
    config.add_argument("soft_placement", bool, required=False, default=True
                        , helps="Whether soft placement is allowed")
    config.add_argument('checkpoint', str, required=False, default=None
                        , helps="Checkpoint model file to load")
    config.add_argument('ckptperbatch', int, required=False, default=2000
                        , helps="Save checkpoint per num minibatch, 0 for no save")
    config.add_argument("max_to_keep", int, required=False, default=5
                        , helps="Maximum number of recent checkpoints to keep.")
    config.add_argument("keep_checkpoint_every_n_hours", int , required=False, default=10
                        , helps=" How often to keep checkpoints.")
    config.add_argument('devicemem', float, required=False, default=1.0
                        , helps="GPU memory fraction to occupy")
    config.add_argument('batchsize', int, required=False, default=100)
    config.add_argument('lr', float, required=False, default=0.0015)
    config.add_argument('momentum', float, required=False, default=0.85)
    config.add_argument('randomrange', float, required=False, default=0.01)
    config.add_argument('maxepoch', int, required=False, default=1
                        , helps="Number of times to go through the data.")
    config.add_argument('chunksize', int, required=False, default=64
                        , helps="Size of chunklist in MB")

    return config.load_file(config_file)


def main():
    if len(sys.argv) == 1:
        print("Usage: train.py --config_fnm --job_name --task_index --deviceid\n"
              "                --config_fnm: INI type configure file which should has 'main' section\n"
              "                --job_name: One of 'ps', 'worker'\n"
              "                --task_index: Index of task within the job\n"
              "                --deviceid: GPU ID: 0-7\n")

        exit(1)

    log = Logging()
    log.message("\n\n")
    log.message("Tensorgou start train work ......\n")
    log.message("======================================================")
    args, inmodule = parseArgs()

    initenviroment(args)
    server, cluster = initDistributed(args)

    if args.isps is True:
        log.message("Parameter server joined!")
        server.join()
    else:
        ## Step 1: Init tensorflow:
        word2id, graph, sess, saver, global_step = initworker(args, server, cluster, inmodule)

        ## Step 3: Init dataset
        dataset = initDataset(args)

        ## Step 4: Run it
        trainLoop(args, inmodule, graph, sess, saver, dataset, word2id, global_step)