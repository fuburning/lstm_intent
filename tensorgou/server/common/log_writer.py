#!/usr/bin/env python27
#encoding=utf-8
import os, sys
reload(sys)
sys.setdefaultencoding("utf-8")

__author__ = "Yuanpeng Zhang"
__date__   = "$2016-11-22$"


class LogWriter(object):
    def __init__(self, filename, mode="w"):
        self.writer = open(filename, mode)
        pass

    def __del__(self):
        self.writer.close()
        pass

    def write(self, logstr):
        self.writer.write(logstr)
        pass

    def writeline(self, logstr):
        self.writer.write(logstr + "\n")
        pass

    def writelines(self, logstrs):
        self.writer.writelines(logstrs)
        pass
