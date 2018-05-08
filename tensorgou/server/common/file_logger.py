#!/usr/bin/env python2.7
#encoding=utf-8
import os, sys
reload(sys)
sys.setdefaultencoding("utf-8")

import logging
import logging.handlers

__author__ = "Yuanpeng Zhang"
__date__   = "$2016-11-22$"


class RotatingFileLogger(logging.Logger):
    """RotatingFileLogger"""

    def __init__(self, filename, level=logging.INFO, **kwargs):
        handler = logging.handlers.RotatingFileHandler(filename, **kwargs)
        handler.setLevel(level)
        fmt = "[%(asctime)s] [%(levelname)s] [%(module)s:%(lineno)s %(funcName)s()] %(message)s"
        datefmt = "%Y-%m-%d %H:%M:%S"
        formater = logging.Formatter(fmt=fmt, datefmt=datefmt)
        handler.setFormatter(formater)

        logging.Logger.__init__(self, "", level)
        self.addHandler(handler)
        pass
