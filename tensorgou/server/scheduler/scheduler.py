#!/usr/bin/env python2.7
#encoding=utf-8
import os, sys
reload(sys)
sys.setdefaultencoding("utf-8")

import threading
import time
import traceback
import functools

from tensorgou.server.scheduler.job import Job

from SimpleXMLRPCServer import SimpleXMLRPCServer
from SocketServer import ThreadingMixIn

import tensorgou.server.scheduler.syspath_autofiller
from tensorgou.server.common import json_util
from tensorgou.server.common import rcode
import tensorgou.server.alert as alert

import tensorgou.server.scheduler.log as log
logger = log.getLogger()


class ThreadingXMLRPCServer(ThreadingMixIn, SimpleXMLRPCServer): pass


def _safe_rpc_wrapper(f):
    @functools.wraps(f)
    def _f(*a, **kw):
        try:
            r, m, v = f(*a, **kw)
            return { "retcode":r, "message":m, "result":v }
        except Exception as e:
            logger.error("rpc function exception: [function:%s] [exception:%s]\n%s" %
                    (f.__name__, str(e), traceback.format_exc()))
            return { "retcode":rcode.RC_EXCEPTION, "message":"Exception:%s" % str(e), "result":{} }
    return _f


class Scheduler(object):
    """Scheduler

    Manage resources and jobs
    """
    def __init__(self, environ):
        self._environ = environ
        self._jobs = list()
        self._jobs_lock = threading.Lock()
        pass

    def _remove_job(self, job):
        with self._jobs_lock:
            self._jobs.remove(job)
        pass

    def _alert(self, job, status):
        notify_sms = job.job_desc.get("notify_sms", "").strip()
        notify_mail = job.job_desc.get("notify_mail", "").strip()
        content = "[tensorgou] job %s [job_guid:%s] [job_name:%s]" % (status, job.job_guid, job.job_name)
        if notify_sms:
            appid = self._environ["conf"]["alert_sms_appid"]
            succ, message = alert.send_message(appid=appid, number=notify_sms, desc=content)
            if not succ:
                logger.error("sms failed [message:%s]" % message)
        if notify_mail:
            fr_name = self._environ["conf"]["alert_mail_from_name"]
            fr_addr = self._environ["conf"]["alert_mail_from_addr"]
            succ, message = alert.send_mail(uid=fr_addr, fr_name=fr_name, fr_addr=fr_addr,
                    title=content, body=content, mail_list=[notify_mail])
            if not succ:
                logger.error("mail failed [message:%s]" % message)
        pass

    def schedule(self):
        with self._jobs_lock:
            jobs = self._jobs[:]

        for job in jobs:
            if job.job_status == Job.ST_SUCCEED:
                logger.info("job succeed [job_guid:%s]" % job.job_guid)
                self._remove_job(job)
                # todo: sms alert
                self._alert(job, "succeed")
            elif job.job_status == Job.ST_FAILED:
                logger.info("job failed [job_guid:%s]" % job.job_guid)
                self._remove_job(job)
                # todo: sms alert
                self._alert(job, "failed")
            elif job.job_status == Job.ST_CANCELED:
                logger.info("job canceled [job_guid:%s]" % job.job_guid)
                self._remove_job(job)
                # todo: sms alert
                self._alert(job, "canceled")
            else:
                try:
                    job.schedule()
                except Exception as e:
                    logger.error("schedule job exception: [job_guid:%s] [exception:%s]\n%s" % (job.job_guid, str(e), traceback.format_exc()))
                    logger.error("schedule job failed, cancel it [job_guid:%s]" % job.job_guid)
                    job.cancel()
                    pass

        del jobs[:]
        pass
    
    @_safe_rpc_wrapper
    def update_status(self, job_guid, task_guid, retcode):
        """update_status(job_guid, task_guid, retcode)"""
        logger.info("rpc update_status [job_guid:%s] [task_guid:%s] [retcode:%d]" % (job_guid, task_guid, retcode))
        with self._jobs_lock:
            for job in self._jobs:
                if job_guid == job.job_guid:
                    job.update_status(task_guid, retcode)
                    return rcode.RC_SUCCEED, "", {}
        logger.error("rpc update_status, can't find job [job_guid:%s]" % job_guid)
        return rcode.RC_FAILED, "can't find job", {}

    @_safe_rpc_wrapper
    def submit_job(self, job_json_string):
        """submit_job(job_json_string)"""
        logger.info("rpc submit_job [job_json_string:%s]" % job_json_string)
        job_params = json_util.LoadString(job_json_string)
        if job_params == None:
            logger.error("rpc submit_job, json format error [job_json_string:%s]" % job_json_string)
            return rcode.RC_FAILED, "json format error", {}
        
        try:
            job = Job(job_params, self._environ)
        except Exception as e:
            logger.error("construct job failed [exception:%s]\n%s" % (str(e), traceback.format_exc()))
            return rcode.RC_FAILED, "construct job failed", {}

        with self._jobs_lock:
            self._jobs.append(job)
        return rcode.RC_SUCCEED, "", { "job_guid": job.job_guid }

    @_safe_rpc_wrapper
    def cancel_job(self, job_guid):
        """cancel_job(job_guid)"""
        logger.info("rpc cancel_job [job_guid:%s]" % job_guid)
        with self._jobs_lock:
            for job in self._jobs:
                if job.job_guid == job_guid:
                    job.cancel()
                    return rcode.RC_SUCCEED, "", {}
        return rcode.RC_FAILED, "can't find job", {}

    @_safe_rpc_wrapper
    def dump_jobs(self):
        """dump_jobs()"""
        logger.info("rpc dump_jobs")
        tmp_list = []
        with self._jobs_lock:
            for job in self._jobs:
                tmp_list.append(job.dump())
        return rcode.RC_SUCCEED, "", { "jobs": tmp_list }

    @_safe_rpc_wrapper
    def dump_job(self, job_guid):
        """dump_job(job_guid)"""
        logger.info("rpc dump_job [job_guid:%s]" % job_guid)
        with self._jobs_lock:
            for job in self._jobs:
                if job.job_guid == job_guid:
                    return rcode.RC_SUCCEED, "", { "job_detail": job.dump_detail() }
        return rcode.RC_FAILED, "can't find job", {}


class SchedulerServer(object):
    """SchedulerServer

    Start rpc server, handle requests about jobs
    """
    def __init__(self, conf):
        self._conf = conf

        import storage
        self._storage = eval(conf["storage"])
        # self._storage = MysqlStorage(conf["mysql_host"], conf["mysql_port"],
                # conf["mysql_user"], conf["mysql_passwd"], conf["mysql_db"])
        self._environ = { "storage": self._storage, "conf": self._conf }
        self._scheduler = Scheduler(self._environ)

        self._listen_host = conf.get("listen_host", "0.0.0.0")
        self._listen_port = int(conf.get("listen_port", 54199))
        self._listen_addr = (self._listen_host, self._listen_port)

        self._rpc_server = ThreadingXMLRPCServer(self._listen_addr, logRequests=False, allow_none=True)
        self._rpc_server.daemon_threads = True
        self._rpc_server.register_introspection_functions()
        self._rpc_server.register_function(self._scheduler.update_status, "update_status")
        self._rpc_server.register_function(self._scheduler.submit_job, "submit_job")
        self._rpc_server.register_function(self._scheduler.cancel_job, "cancel_job")

        self._quit = False
        pass

    def __del__(self):
        self._storage.close()
        pass

    def _schedule_tf(self):
        logger.info("schedule thread started")
        while not self._quit:
            self._scheduler.schedule()
            time.sleep(int(self._conf.get("schedule_interval", 15)))
        logger.info("schedule thread exited")
        pass

    def _rpc_tf(self):
        logger.info("rpc thread started")
        logger.info("rpc server starting [host:%s] [port:%d]" % (self._listen_host, self._listen_port))
        try:
            self._rpc_server.serve_forever()
        except Exception as e:
            logger.error("Exception: %s\n%s" % (str(e), traceback.format_exc()))
        logger.info("rpc thread exited")
        pass

    def start(self):
        logger.info("scheduler started")

        schedule_thread = threading.Thread(target=self._schedule_tf)
        schedule_thread.setDaemon(True)
        schedule_thread.start()

        rpc_thread = threading.Thread(target=self._rpc_tf)
        rpc_thread.setDaemon(True)
        rpc_thread.start()

        # Glances in the past 500 years, an encounter this life. ~.~ . Not supported. #
        while not self._quit:
            time.sleep(500*365*24*60*60 / 500)
        
        logger.info("scheduler exited")
        pass

    def stop(self):
        logger.info("stop scheduler")
        self._quit = True
        self._rpc_server.shutdown()
        pass

