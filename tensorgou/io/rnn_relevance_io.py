"""
Load txt file format defined as:
query1 \t title1 \t title1
query2 \t title2 \t title2
query3 \t title3 \t title3
...
"""
from tensorgou.utils.txtutils import cleanline

__author__ = "Yuanpeng Zhang"
__date__   = "$2016-10-19$"


def doProcess(recodes):
    querys = []
    titles_a = []
    titles_b = []

    for i in range(len(recodes)):
        line = recodes[i]
        line = cleanline(line)
        parts = line.split('\t')
        if not len(parts) == 3:
            raise Exception("Bad line, expect 3 parts [query title1 title2], get {} parts!"
                            .format(len(parts)))
        querys.append(parts[0])
        titles_a.append(parts[1])
        titles_b.append(parts[2])

    return querys, titles_a, titles_b