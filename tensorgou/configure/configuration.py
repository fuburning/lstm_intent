
import codecs
import traceback
from argparse import Namespace
from tensorgou.utils.moduleutils import loadmodel
from tensorgou.utils.txtutils import to_lowercase
from tensorgou.logging import message
from tensorgou.configure.config_loader import load_config_file

__author__ = "Yuanpeng Zhang"
__date__   = "$2016-10-13$"


def setParameter(args, name, value, type, force=False):
    if name not in args.__dict__:
        args.__dict__[name] = value
    else:
        if force is False:
            raise Exception("Parameter '{}' already exist!".format(name))
        if not isinstance(args.name, type):
            raise Exception("Parameter '{}' value '{}' can't match to exist type!"
                            .format(name, args.name))

        args.__dict__[name] = value


class Configuration(object):
    """
    Loads the configuration file in an analogical way the python's
    argparse.ArgumentParser works.
    """

    def __init__(self):
        self.data_types = {}
        self.defaults = {}
        self.conditions = {}
        self.helps = {}
        self.ignored = set()
        self.inmodule = None

    def getmodule(self, config_dict):
        if self.inmodule is not None:
            return

        if not config_dict.has_key('name'):
            raise Exception("No parameter 'name' be defined?")

        ## only support default type here!
        if config_dict.has_key('type'):
            raise Exception("Only support default 'type' define in current version!")
        if not self.defaults.has_key('type'):
            raise Exception("No parameter 'type' be defined [default]?")

        name = to_lowercase(config_dict['name'])
        type = to_lowercase(self.defaults['type'])
        mnm = "tensorgou.graph" + "." + name + "." + type
        message("Import module {}".format(mnm))

        self.inmodule = loadmodel(mnm)
        return self.inmodule

    def inmodule(self):
        assert self.inmodule is not None
        return self.inmodule

    def add_argument(self, name, arg_type=object, required=False, default=None,
                     cond=None, helps=None):

        if name in self.data_types:
            raise Exception("Data filed defined multiple times.")
        self.data_types[name] = arg_type
        if not required:
            self.defaults[name] = default
        if cond is not None:
            self.conditions[name] = cond

        self.helps[name] = helps

    def ignore_argument(self, name):
        self.ignored.add(name)

    def load_file(self, path):
        message("Loading INI file: '{}'".format(path))

        try:
            # config_f = codecs.open(path, 'r', 'utf-8')
            arguments = Namespace()

            config_dict = load_config_file(path, self.ignored)
            self.getmodule(config_dict)
            self.buildparameter()

            self._check_loaded_conf(config_dict)

            for name, value in config_dict.items():
                if name in self.conditions and not self.conditions[name](value):
                    cond_code = self.conditions[name].__code__
                    cond_filename = cond_code.co_filename
                    cond_line_number = cond_code.co_firstlineno
                    raise Exception(
                        "Value of field '{}' does not satisfy "
                        "condition defined at {}:{}."
                        .format(name, cond_filename, cond_line_number))

                setattr(arguments, name, value)

            for name, value in self.defaults.items():
                if name not in arguments.__dict__:
                    arguments.__dict__[name] = value
            message("INI file loaded.")

        except Exception as exc:
            message("Failed to load INI file: {}".format(exc))
            traceback.print_exc()
            exit(1)

        self.printparameter(arguments)

        return arguments, self.inmodule

    def printparameter(self, args):
        message("Parameter Info:")
        message("===============")
        for name in args.__dict__:
            message("{} = {}".format(name, args.__dict__[name]))

        message("===============")

    def buildparameter(self):
        try:
            self.inmodule.getconfigure(self)
        except Exception:
            raise Exception("Module '{}' has no method 'getconfigure' defined?.".format(mnm))

    def _check_loaded_conf(self, config_dict):
        """ Checks whether there are unexpected or missing fields """
        expected_fields = set(self.data_types.keys())

        expected_missing = []
        for name in expected_fields:
            if name not in self.defaults and name not in config_dict:
                expected_missing.append(name)
        if expected_missing:
            raise Exception("Missing mandatory fileds: {}"
                            .format(", ".join(expected_missing)))

        unexpected = []
        for name in config_dict:
            if name not in expected_fields:
                unexpected.append(name)
        if unexpected:
            raise Exception("Unexpected fields: {}"
                            .format(", ".join(unexpected)))

        """ Check data types method """
        for name in config_dict:
            if not isinstance(config_dict[name], self.data_types[name]):
                raise Exception("Bad data type: param {}, expected {}"
                                . format(name, self.data_types[name]))

        self.inmodule.validcheck(config_dict)
