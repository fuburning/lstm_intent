"""
Load txt file format defined as:
label \t sentence
label \t sentence
label \t sentence
...

sentence: word word word
label: label #[zhangyuanpeng] only support one label!
"""
from tensorgou.utils.txtutils import cleanline

__author__ = "Yuanpeng Zhang"
__date__   = "$2016-11-14$"


def doProcess(recodes):
    sentencelist = []
    labellist = []

    for i in range(len(recodes)):
        line = recodes[i]
        line = cleanline(line)
        parts = line.split('\t')
        if not len(parts) == 2:
            raise Exception("Bad line, expect 2 parts [sentence \t label], get {} parts!"
                            .format(len(parts)))
        #modify by xjk 
        sentencelist.append(parts[0])
        labellist.append(parts[1])

    return sentencelist, labellist