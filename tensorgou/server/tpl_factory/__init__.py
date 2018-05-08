#!/usr/bin/env python2.7
#encoding=utf-8
import os, sys
reload(sys)
sys.setdefaultencoding("utf-8")

import threading

_tpl_lock = threading.Lock()
_tpl_times = dict()

def _is_file_changed(filename):
    try:
        new_time = os.stat(filename).st_mtime
        old_time = _tpl_times.get(filename, None)
        if old_time is None:
            _tpl_times[filename] = new_time
        elif new_time > old_time:
            return True
    except Exception:
        return False
    return False

def factory(path):
    if not path.startswith("tpl_factory."):
        return None

    with _tpl_lock:
        module = sys.modules.get(path, None)
        if module is None:
            try: module = __import__(path, fromlist=[""])
            except Exception: return None

        filename = getattr(module, "__file__", None)
        if filename:
            if filename[-4:] in ('.pyo', '.pyc'):
                filename = filename[:-1]
            if _is_file_changed(filename):
                reload(module)
        return sys.modules.get(path, None)

