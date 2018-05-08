#!/usr/bin/env python2.7
#encoding=utf-8
import os, sys
reload(sys)
sys.setdefaultencoding("utf-8")

import copy
import time
import threading

import tensorgou.server.scheduler.syspath_autofiller
from tensorgou.server.common import json_util
from tensorgou.server.common import rcode
from tensorgou.server.common.guid import JobId
from tensorgou.server.scheduler.task import Task

import tensorgou.server.scheduler.log as log
logger = log.getLogger()


class Job(object):
    """Job, set of tasks"""
    
    # job status
    STATUS_INTEGER = [ ST_WAITING, ST_RUNNING, ST_SUCCEED, ST_FAILED, ST_CANCELED ] = [ 0, 1, 2, 3, 4 ]
    STATUS_STRINGS = [ "waiting", "running", "succeed", "failed", "canceled" ]
    
    def __init__(self, job_desc, environ):
        self.job_desc = job_desc
        self.environ = environ
        
        ji = JobId()
        self.job_guid = ji.string_id()
        
        self.job_name = job_desc.get("job_name", "")
        self.job_all_stage = int(job_desc.get("job_task_num", 0))
        self.job_stage = 0
        self.job_begin_time = ""
        self.job_end_time = ""
        self.job_status = Job.ST_WAITING

        # check
        if self.job_all_stage > 0 and len(job_desc.get("job_tasks", [])) != self.job_all_stage:
            raise Exception("job format error!")

        # have no choice but to ... (interact with web)
        if "job_id" in job_desc:
            self.job_id = job_desc["job_id"]
            self.environ["storage"].update_job(self._job_status_tuple())
        else:
            self.job_id = 0
            self.environ["storage"].insert_job(self._job_status_tuple())
            s, r = self.environ["storage"].job_guid_to_job_id(self.job_guid)
            if not s:
                raise Exception("job_guid_to_job_id failed")
            self.job_id = r

        self._job_replacer = copy.deepcopy(job_desc.get("replacer", {}))
        self._job_replacer["__job_guid__"] = self.job_guid
        self._job_replacer["__job_id__"] = self.job_id
        self._job_replacer["__job_name__"] = self.job_name
        self._resource_replacer = {}
        self._tasks = []
        self._lock = threading.Lock()
        self._canceled = False
        pass

    def _job_status_tuple(self):
        # (job_id, job_guid, job_name, job_all_stage, job_stage, job_status, job_brief)
        job_brief_tmp = {
                "begin_time": self.job_begin_time,
                "end_time": self.job_end_time
                }
        succ, job_brief = json_util.DumpString(job_brief_tmp)
        if not succ:
            job_brief = ""
        return (self.job_id, self.job_guid, self.job_name, self.job_all_stage, \
                self.job_stage, self.job_status, job_brief)

    def _finish_job(self, finish_status):
        self.job_end_time = time.strftime("%Y-%m-%d %H:%M:%S")
        self.job_status = finish_status
        logger.info("job finished [job_guid:%s] [job_status:%s]" % (self.job_guid, Job.STATUS_STRINGS[self.job_status]))
        self.environ["storage"].update_job(self._job_status_tuple())
        self.environ["rmgr"].batch_release([ r["__guid__"] for r in self._resource_replacer.values() ])
        self._resource_replacer.clear()
        pass

    def _schedule_task(self, task):
        rc = task.do_pre()
        if rc == rcode.RC_SUCCEED:
            task.do_run()
        elif rc == rcode.RC_FAILED:
            task.do_finish(rcode.RC_FAILED, with_post=False)
        return rc

    def _schedule(self):
        if self.job_status in [ Job.ST_SUCCEED, Job.ST_FAILED, Job.ST_CANCELED ]:
            return
        
        if self.job_status == Job.ST_WAITING:
            self.job_begin_time = time.strftime("%Y-%m-%d %H:%M:%S")
            self.job_status = Job.ST_RUNNING

        if len(self._tasks) == 0:
            if self.job_stage >= self.job_all_stage:
                self._finish_job(Job.ST_SUCCEED)
                return
            if self._canceled:
                self._finish_job(Job.ST_CANCELED)
                return
            task_desc = self.job_desc["job_tasks"][self.job_stage]
            task_nums = int(task_desc.get("amount", 1))
            for i in range(task_nums):
                self._tasks.append(Task(self, task_desc, self.job_stage, i))
            self.environ["storage"].update_job(self._job_status_tuple())
        
        count_array = [ 0 ] * len(Task.STATUS_INTEGER)
        count_all_flag = True
        for t in self._tasks:
            count_array[t.task_status] += 1
            if (not self._canceled) and count_array[Task.ST_FAILED] == 0 and t.task_status == Task.ST_WAITING:
                if self._schedule_task(t) != rcode.RC_SUCCEED:
                    count_all_flag = False
                    break

        if count_all_flag:
            if count_array[Task.ST_SUCCEED] == len(self._tasks):
                del self._tasks[:]
                self.job_stage += 1
            elif count_array[Task.ST_RUNNING] == 0:
                if count_array[Task.ST_FAILED] > 0:
                    self._finish_job(Job.ST_FAILED)
                elif self._canceled:
                    self._finish_job(Job.ST_CANCELED)
        pass

    def schedule(self):
        with self._lock:
            self._schedule()
        pass

    def cancel(self):
        logger.info("job cancel [job_guid:%s]" % self.job_guid)
        with self._lock:
            if self.job_status not in [ Job.ST_SUCCEED, Job.ST_FAILED, Job.ST_CANCELED ]:
                self._canceled = True
        pass

    def update_status(self, task_guid, retcode):
        logger.info("job update_status [job_guid:%s] [task_guid:%s] [retcode:%d]" % (self.job_guid, task_guid, retcode))
        with self._lock:
            for t in self._tasks:
                if t.task_guid == task_guid:
                    t.do_finish(retcode)
                    return
        logger.error("job update_status, can't find task [job_guid:%s] [task_guid:%s]" % (self.job_guid, task_guid))
        pass

    def dump(self):
        """debug only"""
        with self._lock:
            return self._job_status_tuple()

    def dump_detail(self):
        """debug only"""
        with self._lock:
            basic = self._job_status_tuple()
            rtime = []
            for task in self._tasks:
                rtime.append(task.dump())
        return { "basic": basic, "rtime": rtime }



