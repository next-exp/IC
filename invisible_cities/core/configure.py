"""Configure running options for the cities
JJGC August 2016
"""
import argparse
import sys
import os

from .        log_config      import logger
from . import system_of_units as     units


parser = argparse.ArgumentParser()
parser.add_argument("-c", '--config-file',     type=str,  help="configuration file", required=True)
parser.add_argument("-i", '--files-in',        type=str,  help="input file")
parser.add_argument("-o", '--file-out',        type=str,  help="output file")
parser.add_argument("-n", '--nevents',         type=int,  help="number of events to be processed")
parser.add_argument("-f", '--first-event',     type=int,  help="event number for first event")
parser.add_argument("-r", '--run-number',      type=int,  help="run number")
parser.add_argument("-s", '--skip',            type=int,  help="number of events to be skipped", default=0)
parser.add_argument("-p", '--print_mod',       type=int,  help="print every this number of events")
parser.add_argument("-v", dest='verbosity', action="count",                 help="verbosity level", default=0)
parser.add_argument("--run-all",     action="store_true", help="number of events to be skipped") # TODO fix help
parser.add_argument('--show-config', action='store_true', help="do not run")


def print_configuration(options):
    for key, value in sorted(vars(options).items()):
        print("{0: <22} => {1}".format(key, value))


def configure(input_options=sys.argv):
    program, *args = input_options

    CLI = parser.parse_args(args)
    options = read_config_file(CLI.config_file)

    vars(options).update((opt, val) for opt, val in vars(CLI).items() if val is not None)
    logger.setLevel(vars(options).get("verbosity", "info"))

    print_configuration(options)
    return options


def read_config_file(file_name):
    full_file_name             = os.path.expandvars(file_name)
    read_top_level_config_file = make_config_file_reader()
    whole_config               = read_top_level_config_file(full_file_name)
    return whole_config


def make_config_file_reader():
    builtins = __builtins__.copy()
    builtins.update(vars(units))
    globals_ = {'__builtins__': builtins}
    config = argparse.Namespace()
    def read_included_file(file_name):
        full_file_name = os.path.expandvars(file_name)
        with open(full_file_name, 'r') as config_file:
            exec(config_file.read(), globals_, vars(config))
        return config
    builtins['include'] = read_included_file
    return read_included_file
