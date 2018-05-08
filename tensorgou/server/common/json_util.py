#!/usr/bin/env python27
#encoding=utf-8
import os, sys
reload(sys)
sys.setdefaultencoding("utf-8")

import json
import tensorgou.server.common.file_util as file_util

__author__ = "Yuanpeng Zhang"
__date__   = "$2016-11-22$"


def LoadFile(filepath):
    try:
        jsonObj = json.load(file(filepath, 'r'))
    except:
        return None
    return jsonObj

def DumpFile(filepath, jsonObj):
    try:
        jsonString = json.dumps(jsonObj, indent=2)
        if not file_util.WriteFile(filepath, jsonString):
            return False
    except:
        return False
    return True

def LoadString(jsonString):
    try:
        jsonObj = json.loads(jsonString, strict=False)
    except:
        return None
    return jsonObj

def DumpString(jsonObj):
    try:
        jsonString = json.dumps(jsonObj, indent=2)
    except:
        return False, None
    return True, jsonString
