#!/usr/bin/env python27
#encoding=utf-8
import os, sys
reload(sys)
sys.setdefaultencoding("utf-8")

import logging, logging.config
logging.config.fileConfig(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logger.conf'))


def getLogger():
    return logging.getLogger('executor')
