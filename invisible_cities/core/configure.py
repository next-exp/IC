"""Configure running options for the cities
JJGC August 2016
"""
import argparse
import warnings
import inspect
import numpy  as np
import sys
import os

from os.path import basename

from collections     import namedtuple
from collections     import defaultdict
from collections.abc import MutableMapping
from collections.abc import Mapping  as ABCMapping
from collections.abc import Sequence as ABCSequence

from .        log_config      import logger
from . import system_of_units as     units

from ..types.ic_types import NoneType
from ..types.symbols  import ALL_SYMBOLS
from ..types.symbols  import EventRange

from typing import get_origin
from typing import get_args
from typing import Sequence
from typing import Callable
from typing import Optional
from typing import Mapping
from typing import Union
from typing import Tuple
from typing import Any


def event_range(string):
    try:
        return int(string)
    except ValueError:
        if string not in ALL_SYMBOLS:
            raise ValueError("`--event-range` must be an int, all or last")
        return ALL_SYMBOLS[string]


event_range_help = """<stop> | <start> <stop> | all | <start> last"""

# This type is actually defined by a function in components
# Maybe it should be defined there
EventRangeType   = Union[Tuple[int], Tuple[NoneType], Tuple[int, NoneType], Tuple[NoneType, int], Tuple[int, int]]
OneOrManyFiles   = Union[str, Sequence[str]]

parser = argparse.ArgumentParser()
parser.add_argument('config_file',          type=str,            help="configuration file")
parser.add_argument("-i", '--files-in',     type=str,            help="input file")
parser.add_argument("-o", '--file-out',     type=str,            help="output file")
parser.add_argument("-e", '--event-range',  type=event_range,    help=event_range_help, nargs='*')
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

    options.add_cli((opt, val) for opt, val in vars(CLI).items() if val is not None)
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
    # https://docs.python.org/3/reference/executionmodel.html#builtins-and-restricted-execution
    builtins = __builtins__.copy()
    builtins.update(vars(units))
    builtins.update(ALL_SYMBOLS)
    builtins["np"] = np

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


def type_check(value : Any, type_expected : Any):
    type_got = type(value)
    # SPECIAL CASES
    # Note: match/case cannot be used because `case bool` matches everything.
    if type_expected is Sequence and type_got is np.ndarray: return True

    if type_expected is bool : return  np.issubdtype(type_got, bool       )
    if type_expected is int  : return  np.issubdtype(type_got, np.integer )
    if type_expected is float: return (np.issubdtype(type_got, np.floating) or
                                       np.issubdtype(type_got, np.integer ))

    outer          = get_origin(type_expected)
    is_subscripted = outer is not None
    if not is_subscripted:
        return isinstance(value, type_expected)

    # Unfortunately these two methods don't always return typing objects.
    # That's why the if clauses below use ABC types.

    inner = get_args(type_expected)
    if not inner: # no subscript provided, check only outer type
        return issubclass(type_got, outer)

    if   outer is Union      : return any(type_check(value, t       ) for t    in inner)
    elif outer is ABCSequence: return all(type_check(v    , inner[0]) for v    in value)
    elif outer is tuple      : return all(type_check(v    , t       ) for v, t in zip(value, inner))
    elif outer is ABCMapping : return all(type_check(k    , inner[0]) and
                                          type_check(v    , inner[1]) for k, v in value.items())
    else                     : return isinstance(value, outer)


def compare_signature_to_values( function   : Callable
                               , pos_values : Sequence[Any]
                               ,  kw_values : Mapping[str, Any]):
    """
    Compares the types of the arguments passed to a function with
    those in the function's signature in form of type annotations.
    Only valid for functions where all arguments are given as keyword.

    Parameters
    ----------
    function : Callable
        Function to be checked

    pos_values : Mapping
        Positional arguments for function

     kw_values : Mapping
        Keyword arguments for function

    Raises
    ------
    ValueError : if an argument is missing or has the wrong type

    Warns
    -----
    UserWarning : if any item in values is not used
    """
    def get_name(type_):
        try   : return type_.__name__
        except: return str(type_)

    signature     = inspect.signature(function)
    function_name = function.__name__

    positionals = ( inspect._ParameterKind.POSITIONAL_ONLY
                  , inspect._ParameterKind.POSITIONAL_OR_KEYWORD
                  , inspect._ParameterKind.VAR_POSITIONAL)

    parameters  = dict(signature.parameters.items())
    pos_pars    = list(filter( lambda pair: pair[1].kind in positionals
                             , signature.parameters.items()))

    if len(pos_values) > len(pos_pars):
        msg = (f"The function `{function_name}` received {len(pos_values)} "
               f"positional arguments, but it only accepts {len(pos_pars)}")
        raise ValueError(msg)

    for (name, parameter), value in zip(pos_pars, pos_values):
        type_expected = parameter.annotation
        if not type_check(value, type_expected):
            msg = (f"The function {function_name} expects an argument `{name}` "
                   f"of type `{get_name(type_expected)}` but the received value "
                   f"is of type `{get_name(type(value))}`")
            raise ValueError(msg)

        del parameters[name]

    for name, parameter in parameters.items():
        type_expected = parameter.annotation

        if name not in kw_values:
            if parameter.default is inspect._empty:
                msg = (f"The function `{function_name}` is missing an argument "
                       f"`{name}` of type `{get_name(type_expected)}`")
                raise ValueError(msg)
            else:
                continue

        value = kw_values[name]
        if not type_check(value, type_expected):
            msg = (f"The function {function_name} expects an argument `{name}` "
                   f"of type `{get_name(type_expected)}` but the received value "
                   f"is of type `{get_name(type(value))}`")
            raise ValueError(msg)

    for name in kw_values:
        if name not in signature.parameters:
            msg = f"Argument `{name}` is not being used by `{function_name}`"
            warnings.warn(msg, UserWarning)


def check_annotations(f : Callable):
    """
    Performs type check for the first call to `f`.
    Works only for functions without positional arguments.
    """
    first_time = True
    def checked_f(*args, **kwargs):
        nonlocal first_time
        if first_time:
            first_time = False
            compare_signature_to_values(f, args, kwargs)
        return f(*args, **kwargs)

    return checked_f


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
        # Make the namespace read-only *after* its content has been set
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
    def __delitem__(self, key): del self._data[key]
    def __len__    (self):      return len (self._data)
    def __iter__   (self):      return iter(self._data)

    def add_cli(self, keys_and_vals):
        self.push_file('<command line>')
        for k,v in keys_and_vals:
            self[k] = v
        self.pop_file()

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
            file_name = style_filename(self._file[key])
            print(fmt.format(**locals()))
