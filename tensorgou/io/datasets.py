"""
Tensorgou dataset io module
"""

import tensorgou.logging as log
import time

import multiprocessing as mp
import Queue as lQueue

from tensorgou.utils.txtutils import to_lowercase
from tensorgou.utils.moduleutils import loadmodel
from tensorgou.io.mpreader import readerProc
from tensorgou.io.chunklist import getChunkList

__author__ = "Yuanpeng Zhang"
__date__   = "$2016-10-19$"


def initDataset(arguments):
    assert arguments.trainfnms is not None

    flist, chunklist = getChunkList(arguments)

    return datasets(arguments, flist, chunklist)


class datasets(object):
    def __init__(self, arguments, flist, chunklist):
        self.files = flist
        self.chunklist = chunklist
        assert len(flist) > 0
        assert len(chunklist) > 0

        self.batchsize = arguments.batchsize
        self.maxepoch  = arguments.maxepoch

        self.producer = None
        self.quitEvent = mp.Event()
        self.mpQueue = mp.Queue(10)
        self.isrunning = False

        self.processor = self.createProcessor(arguments)
        self.startProducer()

    def __del__(self):
        self.setstop()

    def get_batch(self):
        if not self.isrunning:
            raise Exception("You should start Producer first!")

        ret = []
        while 1:
            try:
                ret = self.mpQueue.get_nowait()
                if len(ret) < self.batchsize:
                    raise Exception("Loader queue get batch size = {}, expect {}"
                                    .format(len(ret), self.batchsize))

                return self.process(ret)

            except lQueue.Empty:
                if self.quitEvent.is_set():
                    raise StopIteration("Get quit event, IO finished!")
                time.sleep(0.1)
                continue

    def createProcessor(self, arguments):
        name = to_lowercase(arguments.name)

        mnm = "tensorgou.io" + "." + name + "_io"
        log.message("Import module {}".format(mnm))

        return loadmodel(mnm)

    def process(self, recodes):
        return self.processor.doProcess(recodes)

    def setstop(self):
        if not self.isrunning:
            return

        self.quitEvent.set()
        self.producer.join()
        self.isrunning = False

    def startProducer(self):
        if self.isrunning:
            log.warnning("Reader process already started!")
            return
        self.producer = mp.Process(name = "Loader"
                                   , target = readerProc
                                   , args = (self.files
                                             , self.chunklist
                                             , self.batchsize
                                             , self.mpQueue
                                             , self.maxepoch
                                             , self.quitEvent))
        self.producer.start()

        self.isrunning = True


