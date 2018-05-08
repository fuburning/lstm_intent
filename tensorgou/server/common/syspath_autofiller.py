#!/usr/bin/env python27
#encoding=utf-8
import os, sys
reload(sys)
sys.setdefaultencoding("utf-8")


__currpath = os.path.dirname(os.path.abspath(__file__))
__partpath = os.path.dirname(__currpath)
sys.path = list(set(sys.path) | set([__currpath, __partpath]))
