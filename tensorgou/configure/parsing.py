""" Module responsible for INI parsing """

import ConfigParser
import importlib
import re

__author__ = "Yuanpeng Zhang"
__date__   = "$2016-10-17$"

LINE_NUM = re.compile(r"^(.*) ([0-9]+)$")

OBJECT_REF = re.compile(r"^<([a-zA-Z][a-zA-Z0-9_]*)>$")
INTEGER = re.compile(r"^[0-9]+$")
FLOAT = re.compile(r"^[0-9]*\.[0-9]*(e[+-]?[0-9]+)?$")
LIST = re.compile(r"\[([^]]*)\]")
TUPLE = re.compile(r"\(([^]]+)\)")
CLASS_NAME = re.compile(
    r"^class_*[a-zA-Z][a-zA-Z0-9_]*(\._*[a-zA-Z][a-zA-Z0-9_]*)+$")


CONSTANTS = {
    'False': False,
    'True': True,
    'None': None
}


def _keyval_parser_dict():
    return {
        INTEGER: int,
        FLOAT: float,
        CLASS_NAME: _parse_class_name,
        OBJECT_REF: lambda x: "object:" + OBJECT_REF.match(x).group(1),
        LIST: _parse_list,
        TUPLE: _parse_tuple
    }


def _split_on_commas(string):
    """Splits a bracketed string by commas, preserving any commas
    inside brackets."""

    items = []
    char_buffer = []
    openings = []

    for i, char in enumerate(string):
        if char == ',' and len(openings) == 0:
            if len(char_buffer) > 0:
                items.append("".join(char_buffer))
            char_buffer = []
            continue
        elif char == ' ' and len(char_buffer) == 0:
            continue
        elif char == '(' or char == '[':
            openings.append(char)
        elif char == ')':
            if openings.pop() != '(':
                raise Exception('Invalid bracket end ")", col {}.'.format(i))
        elif char == ']':
            if openings.pop() != '[':
                raise Exception('Invalid bracket end "]", col {}.'.format(i))
        char_buffer.append(char)

    if len(char_buffer) > 0:
        items.append("".join(char_buffer))
    return items


def _parse_list(string):
    """ Parses the string recursively as a list """

    matched_content = LIST.match(string).group(1)
    if matched_content == '':
        return []

    items = _split_on_commas(matched_content)
    values = [_parse_value(val) for val in items]
    types = [type(val) for val in values]

    if len(set(types)) > 1:
        raise Exception("List must of a same type, is: {}".format(types))

    return values


def _parse_tuple(string):
    """ Parses the string recursively as a tuple """

    items = _split_on_commas(TUPLE.match(string)[1])
    values = [_parse_value(val) for val in items]

    return tuple(values)


def _parse_class_name(string):
    """
    Parse the string as a module or class name.
    Raises Exception when the class (or module) cannot be imported.
    """

    class_parts = string.split(".")
    class_name = class_parts[-1]

    # TODO should we not assume that everything is from tensorgou?
    module_name = ".".join(["tensorgou"] + class_parts[:-1])

    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        # if the problem is really importing the module
        raise Exception("Type '{}' -> module '{}' does not exist?. \n{}"
                        .format(string, module_name, exc))

    try:
        clazz = getattr(module, class_name)
    except AttributeError as exc:
        raise Exception("Type '{}' -> class '{}' does not exist. \n{}"
                        .format(string, class_name, exc))
    return clazz


def _parse_value(string):
    """ Parses the value recursively according to the tensorgou grammar.

    Arguments:
        string: the string to be parsed
    """

    if string in CONSTANTS:
        return CONSTANTS[string]

    for matcher, parser in _keyval_parser_dict().items():
        if matcher.match(string):
            return parser(string)

    return string


def parse_config(config_file):
    """ Parses an INI file into a dictionary """

    """
    line_numbers = (line.strip() + " " + str(i + 1)
                    if line.strip() != "" else ""
                    for i, line in
                    enumerate(config_file)
                   )
    config = ConfigParser.ConfigParser()
    config.readfp(line_numbers)

    """
    config = ConfigParser.ConfigParser()
    config.read(config_file)

    """
    new_config = {}
    for section in config.sections():
        new_config[section] = {}
        for key in config.options(section):
            match = LINE_NUM.match(config.get(section, key))
            new_config[section][key] = match.group(2), match.group(1)
    """
    return config


def parse_file(config_file):
    """ Parses an INI file into a dictionary """

    parsed_dicts = dict()

    config = parse_config(config_file)

    for section in config.sections():
        parsed_dicts[section] = dict()
        for key, value_string in config.items(section):
            try:
                # TODO: add object define, need parse '<>' object
                value = _parse_value(value_string)
            except Exception:
                raise Exception("Cannot parse value: ", value_string)

            parsed_dicts[section][key] = value

    return parsed_dicts
