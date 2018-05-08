"""
Create word_embedding based on word_list
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorgou.logging as log
import tensorgou.utils.namekey as nk
from tensorgou.utils.txtutils import cleanline
import os
import struct
import click

import numpy as np

__author__ = "Yuanpeng Zhang"
__date__   = "$2016-10-19$"


def getWordembedding(args):
    embeddingnumpy = None
    word2id = None
    checkwordlist = False

    if args.__contains__("checkwordlist"):
        checkwordlist = args.checkwordlist

    if args.buildwordembedding is True:
        load_dict = False
        if args.checkpoint is None:
            load_dict = True

        word = wordembedding(word_list_fnm = args.wordlistfnm
                             , output = args.output
                             , word_dict_fnm = args.worddictfnm
                             , checkwordlist=checkwordlist)

        word2id, embeddingnumpy = word.getwordembedding(wedim = args.embeddingsize
                                                        , loaddict = load_dict)

    return word2id, embeddingnumpy


class wordembedding(object):
    def __init__(self, word_list_fnm, output, word_dict_fnm=None, checkwordlist=False):
        if not os.path.exists(word_list_fnm):
            raise Exception("Cant find word list file {}?".format(word_list_fnm))

        if word_dict_fnm is not None:
            if not os.path.exists(word_dict_fnm):
                raise Exception("Cant find word dict file {}?".format(word_dict_fnm))

        self.checkwordlist = checkwordlist
        self.wdnpfnm = os.path.join(output, nk.sKeywordlistnp)
        self.listfnm = word_list_fnm
        self.dictfnm = word_dict_fnm
        self.listnum = 0
        self.dictnum = 0
        self.worddim = 0

    def load_word_list(self, isload=True):
        assert (os.path.exists(self.listfnm))

        word2id = None
        log.message("Load word list file {}......".format(self.listfnm))
        with open(self.listfnm) as f:
            self.listnum = sum(1 for x in f)

            if not isload:
                return word2id

            f.seek(0, 0)
            log.message("\tFound total {} word list".format(self.listnum))
            wordlist = f.readlines()

            """ split to word dict """
            for i in range(len(wordlist)):
                wordlist[i] = cleanline(wordlist[i])

            if self.checkwordlist is True:
                wordset = set(wordlist)
                for item in wordset:
                    if(wordlist.count(item) > 1):
                        raise Exception("Found multi same word {} {} in wordlist!"
                                        .format(item, wordlist.count(item)))

            word2id = dict(zip(wordlist, range(len(wordlist))))

            if not word2id.has_key("</s>"):
                raise Exception("Expect </s> word in file {}".format(self.listfnm))

        return word2id

    def load_word_dict(self, word2id, isload=True, wedim=None):
        assert (os.path.exists(self.dictfnm))

        eb = None
        log.message("Load word dict file {}......".format(self.dictfnm))

        with open(self.dictfnm) as f:
            self.dictnum, = struct.unpack("i", f.read(4))
            self.worddim, = struct.unpack("i", f.read(4))
            if not isload:
                return

            if wedim is not None:
                if not wedim == self.worddim:
                    raise Exception("Embedding size '{}' can't match file '{}' defined size '{}' dictnum='{}'"
                                    .format(wedim, self.dictfnm, self.worddim, self.dictnum))

            log.message("\tInfo: word_num = {}\tword_dim = {}"
                        .format(self.dictnum, self.worddim))

            if self.dictnum < len(word2id):
                log.warnning("\tword_dict num '{}' < word_list num '{}' ?"
                             .format(self.dictnum, len(word2id)))

            eb = np.zeros((len(word2id), self.worddim), dtype = np.float32)

            pb = click.progressbar(length=self.dictnum, label="Load word dict")
            for i in range(self.dictnum):
                word_len, = struct.unpack("i", f.read(4))
                word_str, = struct.unpack(str(word_len) + "s", f.read(word_len))
                if word2id.has_key(word_str):
                    wordid = word2id[word_str]
                    for j in range(self.worddim):
                        elem_value, = struct.unpack("f", f.read(4))
                        eb[wordid, j] = elem_value
                else:
                    for j in range(self.worddim):
                        _, = struct.unpack("f", f.read(4))

                pb.update(1)

            del pb
            print("")

        return eb

    def getwordembedding(self, wedim, loaddict=True):
        word2id = self.load_word_list(isload = True)

        embeddingnumpy = None
        if loaddict is True:
            if not os.path.exists(self.dictfnm):
                raise Exception("Can't find word dict file '{}'!".format(self.dictfnm))

            embeddingnumpy = self.load_word_dict(word2id = word2id
                                                 , isload = True
                                                 , wedim = wedim)
        else:
            embeddingnumpy = np.zeros((len(word2id), wedim), dtype = np.float32)

        return word2id, embeddingnumpy



