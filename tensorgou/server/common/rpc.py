#!/usr/bin/env python2.7
#encoding=utf-8
import os, sys
reload(sys)
sys.setdefaultencoding("utf-8")

import xmlrpclib

__author__ = "Yuanpeng Zhang"
__date__   = "$2016-11-22$"


class SafeWrapper:
    def __init__(self, func):
        self._func = func
        pass

    def __call__(self, *args, **kwargs):
        try:
            return True, self._func(*args, **kwargs)
        except:
            pass
        return False, None


class SafeServerProxy(xmlrpclib.ServerProxy):
    def __init__(self, *args, **kwargs):
        xmlrpclib.ServerProxy.__init__(self, *args, **kwargs)
        pass

    def __getattr__(self, name):
        return SafeWrapper(xmlrpclib.ServerProxy.__getattr__(self, name))

