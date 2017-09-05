"""Configure running options for the cities
JJGC August 2016
"""
import argparse
import sys
import os

from os.path import basename

from collections     import namedtuple
from collections     import defaultdict
from collections.abc import MutableMapping

from .        log_config      import logger
from . import system_of_units as     units


class All:
    pass

def max_events(string):
    try:
        return int(string)
    except ValueError:
        if string == 'All':
            return All
        raise ValueError('`--n-event-max` must be an int or All')


parser = argparse.ArgumentParser()
parser.add_argument('config_file',          type=str,            help="configuration file")
parser.add_argument("-i", '--files-in',     type=str,            help="input file")
parser.add_argument("-o", '--file-out',     type=str,            help="output file")
parser.add_argument("-n", '--n-events-max', type=max_events,     help="number of events to be processed")
parser.add_argument("-f", '--first-event',  type=int,            help="event number for first event", default=0)
parser.add_argument("-r", '--run-number',   type=int,            help="run number")
parser.add_argument("-p", '--print-mod',    type=int,            help="print every this number of events")
parser.add_argument("-v", dest='verbosity', action="count",      help="increase verbosity level", default=0)
parser.add_argument('--print-config-only',  action='store_true', help='do not run the city')

display = parser.add_mutually_exclusive_group()
parser .add_argument('--hide-config',   action='store_true')
parser .add_argument('--no-overrides',  action='store_true', help='do not show overridden values in --show-config')
display.add_argument('--no-files',      action='store_true', help='do not show config files in --show-config')
display.add_argument('--full-files',    action='store_true', help='show config files with full paths in --show-config')


uninteresting_options = '''hide_config no_overrides no_files full_files print_config_only'''.split()


def configure(input_options=sys.argv):
    program, *args = input_options

    CLI = parser.parse_args(args)
    options = read_config_file(CLI.config_file)

    options.update((opt, val) for opt, val in vars(CLI).items() if val is not None)
    logger.setLevel(options.get("verbosity", "info"))

    return options


def read_config_file(file_name):
    full_file_name             = os.path.expandvars(file_name)
    read_top_level_config_file = make_config_file_reader()
    whole_config               = read_top_level_config_file(full_file_name)
    return whole_config


def make_config_file_reader():
    """Create a config file parser with a new empty configuration.

    Configurations can be spread over multiple files, organized in a
    hierachy in which some files can include othes. This factory
    function creates a fresh config parser, which, given the name of a
    top-level config file, will collect all the settings contained in
    the given file and any files it includes, and return a
    corresponding instance of Configuration.

    """

    builtins = __builtins__.copy()
    builtins.update(vars(units))
    builtins['All'] = All
    globals_ = {'__builtins__': builtins}
    config = Configuration()
    def read_included_file(file_name):
        full_file_name = os.path.expandvars(file_name)
        config.push_file(full_file_name)
        with open(full_file_name, 'r') as config_file:
            exec(config_file.read(), globals_, config)
        config.pop_file()
        return config
    builtins['include'] = read_included_file
    return read_included_file


Overridden = namedtuple('Overridden', 'value file_name')


class ReadOnlyNamespace(argparse.Namespace):

    def __setattr__(self, name, value):
        raise TypeError('''Do not set attibutes in this namespace.

If you really must set a value, do it with the setitem
syntax on the Configuration instance from which you got
this namespace.''')

class Configuration(MutableMapping):

    def __init__(self):
        self._data = {}
        self._file = {}
        self._history = defaultdict(list)
        self._file_stack = ['<none>']

    @property
    def as_namespace(self):
        ns = argparse.Namespace(**self._data)
        # Make the namespace nead-only *after* its content has been set
        ns.__class__ = ReadOnlyNamespace
        return ns

    @property
    def _current_filename(self):
        return self._file_stack[-1]

    def __setitem__(self, key, value):
        if key in self._data:
            self._history[key].append(Overridden(self._data[key], self._file[key]))
        self._data[key] = value
        self._file[key] = self._current_filename

    def __getitem__(self, key): return self._data[key]
    def __delitem__(self, key): raise NotImplementedError
    def __len__    (self):      return len (self._data)
    def __iter__   (self):      return iter(self._data)

    def update(self, *others):
        self._data.update(*others)

    def push_file(self, file_name):
        self._file_stack.append(file_name)

    def pop_file(self):
        self._file_stack.pop()

    def display(self):
        conf = self.as_namespace
        longest = max(self._data, key=len)
        width = len(longest) + 3
        fmt            =  "{key:<%d} = {value:<8} {file_name}"             %  width
        fmt_overridden = "#{key:>%d} = {exval:<8} {file_name}  OVERRIDDEN" % (width - 1)

        def style_filename(file_name):
            if conf.   no_files: return ''
            if conf. full_files: return '# ' +          file_name
            return                      '# ' + basename(file_name)

        for key, value in sorted(self._data.items()):
            value = str(value)
            if key in uninteresting_options:
                continue
            if not conf.no_overrides:
                for exval, file_name in self._history[key]:
                    exval = str(exval)
                    file_name = style_filename(file_name)
                    print(fmt_overridden.format(**locals()))
            file_name = style_filename(self._file.get(key, '<command line>'))
            print(fmt.format(**locals()))
