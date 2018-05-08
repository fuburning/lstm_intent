from tensorgou.utils.txtutils import cleanline
import tensorgou.logging as log
from tensorgou.utils.flock import FLOCK
import tensorgou.utils.namekey as nk

import math
import time
import os

__author__ = "Yuanpeng Zhang"
__date__   = "$2016-10-23$"


def filterfiles(files):
    assert not len(files) == 0

    badfiles = []
    goodfiles = []
    for i in range(len(files)):
        filenm = cleanline(files[i])
        if not os.path.exists(filenm):
            badfiles.append(filenm)
        else:
            goodfiles.append(filenm)

    if not len(badfiles) == 0:
        log.error("Can't find data files:")
        for i in range(len(badfiles)):
            log.error("\t'{}'".format(badfiles[i]))

    if len(goodfiles) == 0:
        raise Exception("No available data files!")

    log.message("Data files: ")

    readablefiles = []
    for i in range(len(goodfiles)):
        curflen = os.path.getsize(goodfiles[i])
        curflen /= (1024 * 1024)
        if curflen < 10:
            log.warnning("File: {} {}M, too small?".format(goodfiles[i], curflen))
        else:
            readablefiles.append(goodfiles[i])
            log.message("\tNo.{} file:{}".format(i + 1, goodfiles[i]))
        """
        with open(goodfiles[i]) as f:
            lines = sum(1 for x in f)
            if lines == 0:
                log.warnning("File: {} is empty?".format(goodfiles[i]))
            else:
                totallines += lines
                log.message("\tNo.{} file:{} with {} recodes"
                            .format(i + 1, goodfiles[i], lines))

                readablefiles.append(goodfiles[i])
        """
    if len(readablefiles) == 0:
        raise Exception("Found 0 records?")

    return readablefiles


def doCreateChunkList(arguments):
    chunkfnm = os.path.join(arguments.output, nk.sKeychunklist)
    chunkfnminfo = chunkfnm + nk.sKeychunklistinfo
    assert not os.path.exists(chunkfnm)
    log.message("Build train dataset chunklist {} ...".format(chunkfnm))

    trainfiles = arguments.trainfnms.split(',')
    if len(trainfiles) == 0:
        raise Exception("No train data file be defined?")
    trainfilelist = filterfiles(trainfiles)

    fileid = []
    filelen = []
    for i in range(len(trainfilelist)):
        log.message("No.{} file {} ......".format(i + 1, trainfilelist[i]))
        curflen = os.path.getsize(trainfilelist[i])
        curflen /= 1024 * 1024
        fileid.append(trainfilelist[i])
        #modify by xjk
        #nchunk = int(math.ceil(curflen / 64)) # chunk size == 64M
        nchunk = int(math.ceil(curflen / 1))
        for num in range(nchunk):
            #modify by xjk
            #offset = 64 * num
            offset = 1 * num
            """
            if num == nchunk - 1:
                rest = curflen - offset
                assert rest <= 64
                assert rest > 0
                filelen.append("{}\t{}".format(i, offset * 1024 * 1024))
            else:
            """
            filelen.append("{}\t{}".format(i, offset * 1024 * 1024))
    """
    Write out info
    """
    log.message("Write chunk list file info {} ...".format(chunkfnminfo))
    with open(chunkfnminfo, "w") as f:
        for item in fileid:
            f.write("%s\n" % item)

    log.message("Write chunk list file {} ...".format(chunkfnm))
    with open(chunkfnm, "w") as f:
        ## f.write("%s\n" % len(filelen))
        for item in filelen:
            f.write("%s\n" % item)


def createChunkList(arguments):
    chunklckfnm = os.path.join(arguments.output, nk.sKeychunklistlck)
    chunkfnm = os.path.join(arguments.output, nk.sKeychunklist)
    lck = FLOCK(chunklckfnm)
    while 1:
        ret = lck.lock()
        if ret:
            if not os.path.exists(chunkfnm):
                try:
                    doCreateChunkList(arguments)
                    lck.unlock()
                    return
                except Exception, e:
                    lck.unlock()
                    raise Exception("Can't create file {}?\n{}".format(chunkfnm, e))
            else:
                lck.unlock()
                return

        else:
            time.sleep(1)
            continue


def loadChunkList(chunkfnm, workid, worknum):
    log.message("Load chunk list from file {} ......".format(chunkfnm))
    finfo = chunkfnm + nk.sKeychunklistinfo
    if (not os.path.exists(finfo)) or (not os.path.exists(chunkfnm)):
        raise Exception("Cant find file {} or {}?".format(chunkfnm, finfo))

    """
    Load file info
    """
    flist = None
    chunklist = None
    with open(finfo) as f:
        flist = f.readlines()
    for i in range(len(flist)):
        flist[i] = cleanline(flist[i])

    with open(chunkfnm) as f:
        chunklist = f.readlines()
    """
    Load chunk info
    """

    for i in range(len(chunklist)):
        chunklist[i] = cleanline(chunklist[i])

    assert len(chunklist) > worknum
    nchunkblock = int(math.ceil(len(chunklist) / worknum))
    startid = nchunkblock * workid
    stopid = nchunkblock
    if workid == worknum - 1:
        stopid = len(chunklist) - nchunkblock * (worknum - 1)

    retchunklist = chunklist[startid : startid + stopid - 1]

    return flist, retchunklist


def getChunkList(arguments):
    assert arguments.output is not None

    chunkfnm = os.path.join(arguments.output, nk.sKeychunklist)
    if arguments.distributed is False:
        workid = 0
        worknum = 1
    else:
        workid = arguments.task_index
        worknum = arguments.numworker

    createChunkList(arguments)
    return loadChunkList(chunkfnm, workid, worknum)