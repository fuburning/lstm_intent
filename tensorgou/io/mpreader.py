import tensorgou.logging as log
import time
import os
import random
import Queue as lQueue

__author__ = "Yuanpeng Zhang"
__date__   = "$2016-10-19$"


def loadNextChunk(flist, chunklist, batchsize, ptr):
    ## TODO: load more chunk, do shuffer
    dchunk, ptr = loadChunkData(flist, chunklist, batchsize, ptr)
    dchunk1, ptr = loadChunkData(flist, chunklist, batchsize, ptr)
    assert len(dchunk) >= batchsize
    assert len(dchunk1) >= batchsize
    dchunk.extend(dchunk1)
    random.shuffle(dchunk)

    return dchunk, ptr


def loadChunkData(flist, chunklist, batchsize, ptr):
    count = 0
    while 1:
        indata, ptr = doLoadChunkData(flist, chunklist, ptr)
        ## TODO: if indata size < batch size, need more loader!
        if (indata is None) or (len(indata) < batchsize):
            count += 1
            if count >= len(chunklist):
                raise Exception("Can't get any data from chunk list")
            continue
        else:
            return indata, ptr


def doLoadChunkData(flist, chunklist, ptr):
    if ptr >= len(chunklist):
        ptr = 0
    cl = chunklist[ptr]
    ptr += 1

    info = cl.split('\t')
    assert len(info) == 2
    fileno = int(info[0])
    if fileno >= len(flist):
        raise Exception("Bad chunklist file, expect {} data file, found {} data file"
                        .format(fileno + 1, len(flist)))
    filenm = flist[fileno]
    offset = int(info[1])

    flen = os.path.getsize(filenm)
    if offset >= flen:
        raise Exception("Bad chunklist file, file {} size = {}, expect {}"
                        .format(filenm, flen, offset))

    with open(filenm) as f:
        f.seek(offset, 0)
        readlen = 1 * 1024 * 1024 # want load 64M
        if flen - offset < readlen:
            readlen = flen - offset

        indata = f.read(readlen)

        recodes = indata.split('\n')
        if len(recodes) <= 2:
            return None, ptr

        return recodes[1 : len(recodes) - 1], ptr


def loadNextBatch(buffer, batchsize):
    assert len(buffer) >= batchsize
    ret = []
    if len(buffer) == batchsize:
        return buffer, ret

    return buffer[:batchsize], buffer[batchsize:]


def readerProc(flist, chunklist, batchsize, queue, maxepoch, quitEvent):
    nchunk = len(chunklist)
    assert nchunk > 0
    assert len(flist) > 0

    random.shuffle(chunklist)

    curepoch = 0
    buffer = []
    curptr = 0
    numchunk = 0
    while 1:
        if quitEvent.is_set():
            return
        ## load next batch
        if len(buffer) < batchsize:
            chunkdata, curptr = loadNextChunk(flist, chunklist, batchsize, curptr)
            buffer.extend(chunkdata)
            numchunk += 2
            if numchunk >= nchunk:
                curepoch += 1
                
                if curepoch >= maxepoch:
                    quitEvent.set()
                    return
                random.shuffle(chunklist)
                curptr = 0
                numchunk = 0
            else:
                log.message("Loader: {}/{} chunks loaded ......"
                            .format(numchunk, nchunk))

        batch, buffer = loadNextBatch(buffer, batchsize)

        ## push into queue
        while 1:
            try:
                queue.put_nowait(batch)
                break
            except lQueue.Full:
                if quitEvent.is_set():
                    return
                time.sleep(0.1)
                continue
