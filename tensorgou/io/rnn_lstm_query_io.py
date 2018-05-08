"""
Load txt file format defined as:
query \t answer
query \t answer
query \t answer
...
"""
from tensorgou.utils.txtutils import cleanline

__author__ = "Yuanpeng Zhang"
__date__   = "$2016-10-19$"


def doProcess(recodes):
    querys = []
    answers = []

    for i in range(len(recodes)):
        line = recodes[i]
        line = cleanline(line)
        parts = line.split('\t')
        if not len(parts) == 2:
            raise Exception("Bad line, expect 2 parts [query answer], get {} parts!"
                            .format(len(parts)))
        querys.append(parts[0])
        answers.append(parts[1])

    return querys, answers