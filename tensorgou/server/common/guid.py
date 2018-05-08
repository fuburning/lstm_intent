#!/usr/bin/env python2.7
#encoding=utf-8
import os, sys
reload(sys)
sys.setdefaultencoding("utf-8")

import threading
import time

__author__ = "Yuanpeng Zhang"
__date__   = "$2016-11-21$"


def _millisecond():
    return long(time.time() * 1000)

class Sequence:
    def __init__(self):
        self._lock = threading.Lock()
        self._seq  = 0
        pass

    def next_seq(self):
        self._lock.acquire()
        self._seq += 1
        seq = self._seq
        self._lock.release()
        return seq

class LoopSequence:
    def __init__(self, limit):
        self._lock  = threading.Lock()
        self._seq   = 0
        self._limit = limit
        pass

    def next_seq(self):
        self._lock.acquire()
        self._seq += 1
        if self._seq >= self._limit:
            self._seq = 1
        seq = self._seq
        self._lock.release()
        return seq

class JobId:
    _job_id_str_prefix = "job_"
    _job_sequence = LoopSequence(10000)

    @staticmethod
    def job_id(job_id_string):
        p = job_id_string[len(JobId._job_id_str_prefix):]
        ps = p.split("_")
        if len(ps) != 2: return None
        return JobId(long(ps[0]), int(ps[1]))

    def __init__(self, millisecond=None, seq=None):
        if millisecond == None: millisecond = _millisecond()
        if seq == None: seq = JobId._job_sequence.next_seq()
        self._millisecond, self._seq = millisecond, seq
        pass

    def __eq__(self, o):
        return self._millisecond == o._millisecond \
                and self._seq == o._seq

    def string_id(self):
        return "{0}{1}_{2:0>4}".format(JobId._job_id_str_prefix,
                self._millisecond,
                self._seq)

class TaskId:
    @staticmethod
    def task_id(task_id_string):
        pos1 = task_id_string.rfind("_")
        if pos1 == -1: return None
        pos2 = task_id_string.rfind("_", 0, pos1)
        if pos2 == -1: return None
        p1 = task_id_string[0:pos2]
        p2 = int(task_id_string[pos2+1:pos1])
        p3 = int(task_id_string[pos1+1:])
        return TaskId(p1, p2, p3)

    def __init__(self, job_id_string, t_seq=0, st_seq=0):
        self._job_id_string = job_id_string
        self._t_seq = t_seq
        self._st_seq = st_seq
        pass

    def __eq__(self, o):
        return self._job_id_string == o._job_id_string \
                and self._t_seq == o._t_seq \
                and self._st_seq == o._st_seq

    def string_id(self):
        return "{0}_{1:0>3}_{2:0>6}".format(self._job_id_string,
                self._t_seq,
                self._st_seq)
    
    def string_last_id(self):
        return "{0:0>3}_{1:0>6}".format(self._t_seq, self._st_seq)

class ResourceId:
    _resource_sequence = LoopSequence(10000)
    _resource_id_string_prefix = "res_"

    @staticmethod
    def resource_id(resource_id_string):
        p = resource_id_string[len(ResourceId._resource_id_string_prefix):]
        ps = p.split("_")
        if len(ps) != 2: return None
        return ResourceId(long(ps[0]), int(ps[1]))

    def __init__(self, millisecond=None, seq=None):
        if millisecond == None: millisecond = _millisecond()
        if seq == None: seq = ResourceId._resource_sequence.next_seq()
        self._millisecond, self._seq = millisecond, seq
        pass
    
    def __eq__(self, o):
        return self._millisecond == o._millisecond \
                and self._seq == o._seq

    def string_id(self):
        return "{0}{1}_{2:0>4}".format(
                ResourceId._resource_id_string_prefix,
                self._millisecond,
                self._seq)

