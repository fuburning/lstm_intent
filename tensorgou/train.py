"""
Tensorgou Train script
"""
import sys
import os
import argparse

from tensorgou.logging import Logging
import tensorgou.configure.configuration as configuration
from tensorgou.trainutils import inittensorflow, trainLoop, initenviroment,testLoop
from tensorgou.io.datasets import initDataset

__author__ = "Yuanpeng Zhang"
__date__   = "$2016-10-17$"


def parseArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_fnm', type=str
                        , help="INI type configure file which should has 'main' section")
    args = parser.parse_args()
    return create_config(args.config_fnm)


def create_config(config_file):
    config = configuration.Configuration()
    config.add_argument('name', str, required=True)
    config.add_argument('type', str, required=False, default="train")
    config.add_argument('distributed', bool, required=False, default=False)
    config.add_argument('output', str, required=True
                        , helps="Result save path")
    config.add_argument("num_cores", int, required=False, default=1
                        , helps="Number of CPU cores to be used. Default: All")
    config.add_argument("log_device", bool, required=False, default=False
                        , helps="Log device placement or not")
    config.add_argument("soft_placement", bool, required=False, default=True
                        , helps="Whether soft placement is allowed")
    config.add_argument("max_to_keep", int, required=False, default=5
                        , helps="Maximum number of recent checkpoints to keep.")
    config.add_argument('deviceid', int, required=False, default="0"
                        , helps="0-7 for gpu id")
    config.add_argument('checkpoint', str, required=False, default=None
                        , helps="Checkpoint model file to load")
    config.add_argument('ckptperbatch', int, required=False, default=2000
                        , helps="Save checkpoint per num minibatch, 0 for no save")
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
        print("Usage: train.py --config_fnm\n"
              "                --config_fnm: INI type configure file which should has 'main' section\n")

        exit(1)

    log = Logging()
    log.message("\n\n")
    log.message("Tensorgou start train work ......\n")
    log.message("======================================================")
    args, inmodule = parseArgs()

    initenviroment(args)
    ## Step 1: Init tensorflow:
    #add graph_test by xjk
    word2id, graph, sess, saver,graph_test = inittensorflow(args, inmodule)

    ## Step 3: Init dataset
    dataset = initDataset(args)

    ## Step 4: Run it
    trainLoop(args, inmodule, graph, sess, saver, dataset, word2id)
    
    #add by xjk
    testLoop(args,inmodule,graph_test,sess,saver,word2id)

