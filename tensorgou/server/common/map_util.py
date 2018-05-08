#!/usr/bin/env python27
#encoding=utf-8
import os, sys
reload(sys)
sys.setdefaultencoding("utf-8")

__author__ = "Yuanpeng Zhang"
__date__   = "$2016-11-22$"


def multi_get(m, fields):
    for f in fields:
        if isinstance(m, list):
            m = m[int(f)]
        elif f in m:
            m = m[f]
        else:
            return False, None
    return True, m


def multi_get_string(m, fields):
    succ, obj = multi_get(m, fields)
    if succ:
        return True, str(obj)
    return False, ""
