#!/usr/bin/env python27
#encoding=utf-8
import os, sys
reload(sys)
sys.setdefaultencoding("utf-8")

__author__ = "Yuanpeng Zhang"
__date__   = "$2016-11-21$"


def WriteFile(filepath, content):
    filepath = os.path.abspath(filepath)
    dirpath  = os.path.dirname(filepath)
    try:
        if not os.path.isdir(dirpath):
            os.makedirs(dirpath)
        with open(filepath, "w") as f:
            f.write(content)
    except:
        return False
    return True
