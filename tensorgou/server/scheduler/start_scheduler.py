#!/usr/bin/env python2.7
#encoding=utf-8
import os, sys
reload(sys)
sys.setdefaultencoding("utf-8")

import traceback

import tensorgou.server.scheduler.syspath_autofiller
from tensorgou.server.scheduler.scheduler import SchedulerServer
from tensorgou.server.common import json_util

import tensorgou.server.scheduler.log as log
logger = log.getLogger()

def main(options, args):
    conf = json_util.LoadFile(options.config_file)
    if conf == None:
        logger.error("config_file format error [config_file:%s]" % options.config_file)
        return False
    logger.info("configure is %s" % str(conf))

    server = SchedulerServer(conf)

    import signal
    def sig_handler(number, frame):
        logger.info("receive signal [number: %d]" % number)
        server.stop()
    signal.signal(signal.SIGTERM, sig_handler)
    signal.signal(signal.SIGINT, sig_handler)

    logger.info("tensorgou server start")
    server.start()
    logger.info("tensorgou server exit")
    return True

if __name__ == "__main__":
    usage = "python2.7 %s -c conf.json" % sys.argv[0]

    from optparse import OptionParser
    parser = OptionParser(usage=usage)
    parser.add_option("-c", "--config", dest="config_file", help="config file path")
    (options, args) = parser.parse_args()

    if options.config_file == None:
        parser.print_help()
        exit(1)

    succ = True
    try:
        succ = main(options, args)
    except Exception as e:
        logger.error("Exception: %s\n%s" % (str(e), traceback.format_exc()))
        exit(1)

    if not succ:
        logger.error("Failed to run scripts")
        exit(1)
    exit(0)

