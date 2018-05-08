import multiprocessing
from Queue import Full, Empty

class QueueHandler(object):
    def __init__(self, q, stop_event, time_out = 5, retries = 5):
        self.q = q
        self.time_out = time_out
        self.retries = retries
        self.stop_event = stop_event

    def push(self, obj):
        retries = 0
        success = False
        while not success and not retries >= self.retries:
            if self.stop_event.is_set():
                return 0
            try:
                self.q.put(obj, True, self.time_out)
                success = True
            except Full:
                retries += 1
        if success:
            return retries
        raise Full

    def pop(self):
        retries = 0
        success = False
        obj = None
        while not success and not retries >= self.retries:
            if self.stop_event.is_set():
                return 0
            try:
                obj = self.q.get(True, self.timeout)
                success = True
            except Empty:
                retries += 1
        if success:
            return obj
        raise Empty
