import importlib

__author__ = "Yuanpeng Zhang"
__date__   = "$2016-10-17$"


def loadmodel(modulenm):
    try:
        module = importlib.import_module(modulenm)
    except ImportError as exc:
        # if the problem is really importing the module
        raise Exception("Import module '{}' failed! \n{}".format(modulenm, exc))

    return module