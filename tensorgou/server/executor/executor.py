#!/usr/bin/env python2.7
#encoding=utf-8
import os, sys
reload(sys)
sys.setdefaultencoding("utf-8")

__author__ = "Yuanpeng Zhang"
__date__   = "$2016-11-22$"

from SimpleXMLRPCServer import SimpleXMLRPCServer
import subprocess
import threading
import time
import traceback
import functools

from tensorgou.server.executor import syspath_autofiller
from tensorgou.server.common import file_logger
from tensorgou.server.common import rpc
from tensorgou.server.common import rcode

import tensorgou.server.executor.log as log
logger = log.getLogger()


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


class Executor(object):
    """Executor"""

    def __init__(self, conf):
        self._conf = conf

        self._master_host = conf.get("master_host", "localhost")
        self._master_port = int(conf.get("master_port", 51280))
        self._master_url = "http://%s:%d" % (self._master_host, self._master_port)
        self._rootdir = conf.get("executor_workdir", ".")

        self._quit = False
        self._quit_event = threading.Event()
        pass

    def stop(self):
        self._quit = True
        self._quit_event.set()
        pass
    
    @_safe_rpc_wrapper
    def execute(self, job_guid, task_guid, command):
        """execute(job_guid, task_guid, command)"""

        logger.info("rpc execute [job_guid:%s] [task_guid:%s] [command:%s]" %
                (job_guid, task_guid, command))
        thread = threading.Thread(target=self._execute_tf, args=(job_guid, task_guid, command))
        thread.setDaemon(True)
        thread.start()
        return rcode.RC_SUCCEED, "", {}

    def _execute_tf(self, job_guid, task_guid, command):
        workdir = os.path.join(self._rootdir, job_guid, task_guid)
        rundir = os.path.join(workdir, "rundir")
        timestr = time.strftime("%Y_%m_%d_%H_%M_%S")

        # prepare workdir
        os.system("rm -rf %s; mkdir -p %s" % (workdir, rundir))
        log_filename = "%s/log.%s" % (workdir, timestr)
        out_filename = "%s/out.%s" % (workdir, timestr)
        err_filename = "%s/err.%s" % (workdir, timestr)

        _log = file_logger.RotatingFileLogger(log_filename)
        _out = file(out_filename, "w")
        _err = file(err_filename, "w")

        _log.info("log_filename: %s", log_filename)
        _log.info("out_filename: %s", out_filename)
        _log.info("err_filename: %s", err_filename)
        _log.info("job_guid: %s", job_guid)
        _log.info("task_guid: %s", task_guid)
        _log.info("command: %s", command)

        retcode = rcode.RC_SUCCEED
        try:
            retcode = subprocess.call(command, stdout=_out, stderr=_err,
                    shell=True, env=os.environ.copy(), cwd=rundir)
        except Exception as e:
            retcode = rcode.RC_FAILED
            _log.error("Execption: %s\n%s" % (str(e), traceback.format_exc()))

        _log.info("retcode: %d" % retcode)

        master = rpc.SafeServerProxy(self._master_url)
        while not self._quit:
            succ, result = master.update_status(job_guid, task_guid, retcode)
            if succ:
                _log.info("reply to master succeed")
                break
            _log.error("reply to master failed, try again five seconds later ...")
            self._quit_event.wait(5)
        os.system("rm -rf %s" % rundir)
        pass

class ExecutorServer(object):
    """ExecutorServer"""

    def __init__(self, conf):
        self._conf = conf

        self._listen_host = conf.get("listen_host", "0.0.0.0")
        self._listen_port = int(conf.get("listen_port", 51290))
        self._listen_addr = (self._listen_host, self._listen_port)

        self._executor = Executor(conf)

        self._rpc_server = SimpleXMLRPCServer(self._listen_addr, logRequests=False, allow_none=True)
        self._rpc_server.register_introspection_functions()
        self._rpc_server.register_function(self._executor.execute, "execute")

        self._quit = False
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
        logger.info("executor started")

        rpc_thread = threading.Thread(target=self._rpc_tf)
        rpc_thread.setDaemon(True)
        rpc_thread.start()

        # Glances in the past 500 years, an encounter this life. ~.~ . Not supported. #
        while not self._quit:
            time.sleep(500*365*24*60*60 / 500)
        
        logger.info("executor exited")
        pass

    def stop(self):
        logger.info("stop executor")
        self._executor.stop()
        self._quit = True
        self._rpc_server.shutdown()
        pass

