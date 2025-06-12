from os   import path
from enum import Enum

import numpy as np

from pytest import fixture
from pytest import mark
from pytest import raises
from pytest import warns

from . configure import configure
from . configure import Configuration
from . configure import make_config_file_reader
from . configure import read_config_file
from . configure import type_check
from . configure import compare_signature_to_values
from . configure import check_annotations
from .           import system_of_units  as units

from . exceptions import NoInputFiles
from . exceptions import NoOutputFile

from .. types .symbols     import EventRange
from .. cities.components  import city
from .. cities.penthesilea import penthesilea

from typing import Sequence
from typing import Mapping
from typing import Optional
from typing import Tuple
from typing import List
from typing import Union


all  = EventRange.all
last = EventRange.last


config_file_format = """
# set_input_files
files_in = '{files_in}' # comment to be ignored 1
# set_pmap_store
file_out = '{file_out}'

compression = '{compression}'

# irene
run_number = {run_number}

# set_print
nprint = {nprint}

# set_blr
nbaseline   = {nbaseline}
thr_trigger = {thr_trigger}

# set_maw
n_maw   = {n_maw}
thr_maw = {thr_maw}

# set_csum
thr_csum = {thr_csum}

# set_s1
s1_tmin = {s1_tmin}  # * mus
s1_tmax = {s1_tmax}
s1_stride = {s1_stride}
s1_lmin = {s1_lmin}
s1_lmax = {s1_lmax}

# set_s2
s2_tmin = {s2_tmin}
s2_tmax = {s2_tmax}
s2_stride = {s2_stride}
s2_lmin = {s2_lmin}
s2_lmax = {s2_lmax}

# set_sipm
thr_zs = {thr_zs}
thr_sipm_s2 = {thr_sipm_s2}

# run
event_range = {event_range}
"""

# The values that will be fed into the above.
config_file_spec = dict(files_in    = 'electrons_40keV_z25_RWF.h5',
                        file_out    = 'electrons_40keV_z25_PMP.h5',
                        compression = 'ZLIB4',
                        run_number  = 23,
                        nprint      = 24,
                        nbaseline   = 26,
                        thr_trigger = 27,
                        n_maw       = 28,
                        thr_maw     = 29,
                        thr_csum    = .5,
                        s1_tmin     = 31,
                        s1_tmax     = 32,
                        s1_stride   = 33,
                        s1_lmin     = 34,
                        s1_lmax     = 35,
                        s2_tmin     = 36,
                        s2_tmax     = 37,
                        s2_stride   = 39,
                        s2_lmin     = 41,
                        s2_lmax     = 42,
                        thr_zs      = 43,
                        thr_sipm_s2 = 44,
                        event_range = (45, 46))

config_file_contents = config_file_format.format(**config_file_spec)


def join_dicts(*args):
    """Create new dict by combining any number of dicts.

    When keys appear in more than one dict, the values of the later
    ones override those of the earlier ones.
    """
    accumulate = {}
    for d in args:
        accumulate.update(d)
    return accumulate


def write_config_file(file_name, contents):
    with open(file_name, 'w') as out:
        out.write(contents)


@fixture(scope="session")
def default_conf(config_tmpdir):
    conf_file_name = config_tmpdir.join('test.conf')
    write_config_file(conf_file_name, config_file_contents)
    return conf_file_name


# This test is run repeatedly with different specs. Each spec is a
# sequence of command-line variable specifications. The spec should
# contain one entry for each variable that is being set on the command
# line, consisting of
#
# (<variable name>, <command line arg>, <expected resulting value>)
@mark.parametrize('spec',
                  (
                    # Nothing overridden on the command line
                    (),
                    # Two short form command line args
                    (('run_number',   '-r 99', 99),
                     ('print_mod',    '-p 98', 98)),
                    # A long option in full
                    (('run_number' , '--run-number 97', 97),),
                    # A long option abbreviated
                    (('run_number' , '--ru 6', 6),),
                    # Verbosity level 1
                    (('verbosity', '-v',    1),),
                    # Verbosity level 2
                    (('verbosity', '-v -v', 2),),
                    # Verbosity level 3
                    (('verbosity', '-vvv',  3),),
                    # Verbosity level 4
                    (('verbosity', '-vvv -v ', 4),),
                    # Input and output files
                    (('files_in', '-i some_input_file',  'some_input_file'),
                     ('file_out', '-o some_output_file', 'some_output_file')),
                  ))
def test_configure(default_conf, spec):
    # Extract command line aruments and the corresponding desired
    # values from the test's parameter.
    extra_args   = [    arg       for (_, arg, _    ) in spec ]
    cmdline_spec = { opt : value  for (opt, _, value) in spec }

    # Compose the command line
    args_base = 'program_name {default_conf} '.format(**locals())
    args_options = ' '.join(extra_args)
    args = (args_base + args_options).split()

    # Present the command line to the parser
    CFP = configure(args).as_namespace

    # The values in the configuration can come from three different places
    # 1. Default values
    # 2. The configuration file can override these
    # 3. Command line arguments can override all of these
    this_configuration = join_dicts(config_file_spec,
                                    cmdline_spec)

    # Expand environment variables in paths
    for k,v in this_configuration.items():
        try:
            if '$' in v:
                this_configuration[k] = path.expandvars(v)
        except TypeError:
            pass


    # Check that all options have the expected values
    for option in this_configuration:
        assert getattr(CFP, option) == this_configuration[option], 'option = ' + option


@mark.parametrize("flag", ("-i", "--input-files" ,
                           "-o", "--output-files",
                           "-r", "--run-number"  ,
                           "-p", "--print-mod"   ))
def test_configure_does_not_take_multiple_arguments(default_conf, flag):
    iargs = " ".join(f"arg_{i}.h5" for i in range(2))
    argv   = f"dummy {default_conf} {flag} {iargs}".split()
    with raises(SystemExit):
        configure(argv)


def test_configure_raises_SystemExit_with_multiple_mutually_exclusive_options():
    argv = f"dummy {default_conf} --no-files --full-files".split()
    with raises(SystemExit):
        configure(argv)


def test_read_config_file_special_values(config_tmpdir):
    filename  = path.join(config_tmpdir, "test_read_config_file_special_values.conf")
    all_units = list(vars(units).keys())
    write_config_file(filename, f"""
var_all    = all
var_last   = last
vars_units = {all_units}
""")

    conf = read_config_file(filename)
    assert conf["var_all"   ] is all
    assert conf["var_last"  ] is last
    assert conf["vars_units"] == all_units


@fixture(scope = 'session')
def hierarchical_configuration(tmpdir_factory):
    "Create a dummy city configuration spread across a hierarchy of included files."
    dir_ = tmpdir_factory.mktemp('test_config_files')
    top_file_name = path.join(dir_, 'top_file')
    include_1     = path.join(dir_, 'include_1')
    include_1_1   = path.join(dir_, 'include_1_1')
    include_2     = path.join(dir_, 'include_2')
    write_config_file(top_file_name, """
include('{include_1}')
set_just_once_in_top_file = 1
overridden_in_top_file = 'original'
overridden_in_top_file = 'first override'
overridden_in_top_file = 'second override'
overridden_in_top_file = 'final override'
include_1_overridden_by_top = 'top overrides'
top_overridden_by_include_2 = 'replaced'
include('{include_2}')
overridden_in_3_places = 'four'
""".format(**locals()))

    write_config_file(include_1, """
include('{include_1_1}')
only_in_include_1 = 'just 1'
include_1_overridden_by_top = 'gone'
overridden_in_3_places = 'two'
""".format(**locals()))

    write_config_file(include_1_1, """
only_in_include_1 = 'just 1'
top_overridden_by_include_2 = 'i2 overrides'
overridden_in_3_places = 'one'
""")

    write_config_file(include_2, """
only_in_include_1 = 'just 1'
top_overridden_by_include_2 = 'i2 overrides'
overridden_in_3_places = 'three'
""")

    return make_config_file_reader()(top_file_name)


def test_config_set_just_once_in_top_file(hierarchical_configuration):
    conf = hierarchical_configuration.as_namespace
    assert conf.set_just_once_in_top_file == 1

def test_config_overridden_in_top_file_value(hierarchical_configuration):
    conf = hierarchical_configuration.as_namespace
    assert conf.overridden_in_top_file == 'final override'

def test_config_overridden_in_top_file_history(hierarchical_configuration):
    history = hierarchical_configuration._history['overridden_in_top_file']
    value_history = [ i.value for i in history ]
    assert value_history == ['original', 'first override', 'second override']

def test_config_only_in_include(hierarchical_configuration):
    conf = hierarchical_configuration.as_namespace
    assert conf.only_in_include_1 == 'just 1'

def test_config_include_overridden_by_top(hierarchical_configuration):
    conf = hierarchical_configuration.as_namespace
    assert conf.include_1_overridden_by_top == 'top overrides'

def test_config_top_overridden_by_include(hierarchical_configuration):
    conf = hierarchical_configuration.as_namespace
    assert conf.top_overridden_by_include_2 == 'i2 overrides'

def test_config_overridden_file_history(hierarchical_configuration):
    history = hierarchical_configuration._history['overridden_in_3_places']
    file_history  = [ path.basename(i.file_name) for i in history ]
    value_history = [               i.value      for i in history ]

    assert file_history  == ['include_1_1', 'include_1', 'include_2']
    assert value_history == ['one',         'two',       'three'    ]
    assert hierarchical_configuration.as_namespace.overridden_in_3_places == 'four'


def test_Configuration_missing_key_raises_KE():
    c = Configuration()
    with raises(KeyError):
        c['absent']

def test_Configuration_setitem_getitem():
    c = Configuration()
    c['some_key'] = 'some value'
    assert c['some_key'] == 'some value'

def test_Configuration_as_namespace_getattr():
    c = Configuration()
    c['something'] = 'its value'
    ns = c.as_namespace
    assert ns.something == 'its value'

def test_Configuration_as_namespace_is_read_only():
    c = Configuration()
    ns = c.as_namespace
    with raises(TypeError):
        ns.new_attribute = 'any value'


@fixture(scope = 'session')
def simple_conf_file_name(tmpdir_factory):
    dir_ = tmpdir_factory.mktemp('test_config_files')
    file_name = path.join(dir_, 'simple.conf')
    write_config_file(file_name, """
compression  = 'ZLIB4'
run_number   = 12
print_mod    = 13
event_range  = 14,
""")
    return str(file_name)


@city
def dummy(**kwds):
    pass

def test_config_city_fails_without_config_file():
    argv = 'dummy'.split()
    with raises(SystemExit):
        dummy(**configure(argv))

def test_config_city_fails_without_input_file(simple_conf_file_name):
    argv = 'dummy {simple_conf_file_name}'.format(**locals()).split()
    with raises(NoInputFiles):
        dummy(**configure(argv))

def test_config_city_fails_without_output_file(simple_conf_file_name):
    conf   = simple_conf_file_name
    infile = conf # any existing file will do as a dummy for now
    argv = 'dummy {conf} -i {infile}'.format(**locals()).split()
    with raises(NoOutputFile):
        dummy(**configure(argv))


@mark.parametrize(     'name             flags           value'.split(),
                  (('run_number' ,                 '-r 23', 23),
                   ('run_number' ,       '--run-number 24', 24),
                   ('print_mod'  ,                 '-p 25', 25),
                   ('print_mod'  ,        '--print-mod 26', 26),
                   ('event_range',                '-e all', [all]),
                   ('event_range',     '--event-range all', [all]),
                   ('event_range',                 '-e 27', [27]),
                   ('event_range',     '--event-range  28', [28]),
                   ('event_range',            '-e 29 last', [29, last]),
                   ('event_range', '--event-range 30 last', [30, last]),
                   ('event_range',              '-e 31 32', [31, 32]),
                   ('event_range',   '--event-range 33 34', [33, 34]),
                  ))
def test_config_CLI_flags(simple_conf_file_name, tmpdir_factory, name, flags, value):
    conf   = simple_conf_file_name
    infile = conf # any existing file will do as a dummy for now
    outfile = tmpdir_factory.mktemp('config-flags').join('dummy-output-file' + name)
    argv = 'dummy {conf} -i {infile} -o {outfile} {flags}'.format(**locals()).split()
    conf_ns = configure(argv).as_namespace
    assert getattr(conf_ns, name) == value


def test_configure_numpy(config_tmpdir):
    config_filename = config_tmpdir.join("conf_with_numpy.conf")
    config_contents = "a_numpy_array = np.linspace(0, 1, 3)\n"
    write_config_file(config_filename, config_contents)

    argv    = f"some_city {config_filename}".split()
    conf_ns = configure(argv).as_namespace
    assert hasattr(conf_ns, "a_numpy_array")
    assert conf_ns.a_numpy_array.tolist() == [0, 0.5, 1]


@mark.parametrize( "value type".split()
                 , ( (  0, int)
                   , ( "", str)
                   , ( [], List)
                   , ( [], Sequence)
                   , ( (), Tuple)
                   , ( (), Sequence)
                   , (all, EventRange)))
def test_type_check_simple(value, type):
    assert type_check(value, type)


@mark.parametrize(          "value  type".split()
                 , ( (           0, float)
                   , (np.arange(1), Sequence)))
def test_type_check_special_cases(value, type):
    assert type_check(value, type)


@mark.parametrize( "value type".split()
                 , ( ("", int)
                   , ( 0, str)
                   , ( 0, List)
                   , ( 0, Sequence)
                   , ("", Tuple)
                   , ("", EventRange)))
def test_type_check_wrong_type(value, type):
    assert not type_check(value, type)


@mark.parametrize("value", (1, "", all))
def test_type_check_union(value):
    type = Union[int, str, EventRange]
    assert type_check(value, type)


@mark.parametrize("value", (1.0, (), [], {}))
def test_type_check_union_wrong_type(value):
    type = Union[int, str, EventRange]
    assert not type_check(value, type)


@mark.parametrize("nvalues", (1, 5))
def test_type_check_sequence(nvalues):
    type  = Sequence[int]
    value = list(range(nvalues))
    assert type_check(value, type)


def test_type_check_sequence_wrong_type_all():
    type  = Sequence[str]
    value = list(range(5))
    assert not type_check(value, type)


def test_type_check_sequence_wrong_type_one():
    type  = Sequence[str]
    value = ["0", 1, "2"]
    assert not type_check(value, type)


def test_type_check_tuple():
    type  = Tuple[int, str, EventRange]
    value = (1, "", all)
    assert type_check(value, type)


def test_type_check_tuple_wrong_type():
    type  = Tuple[int, str, EventRange]
    value = (all, "", 1)
    assert not type_check(value, type)


def test_type_check_mapping():
    type  = Mapping[str, int]
    value = dict(a=1)
    assert type_check(value, type)


def test_type_check_mapping_many_values():
    type  = Mapping[str, int]
    value = dict(a=1, b=2, c=3)
    assert type_check(value, type)


def test_type_check_mapping_wrong_type():
    type  = Mapping[str, int]
    value = dict(a="1")
    assert not type_check(value, type)


def test_type_check_optional():
    type  = Optional[int]
    value = 1
    assert type_check(value, type)


@mark.parametrize("value", (1, ("", all)))
def test_type_check_nested_types(value):
    type = Optional[Union[int, Tuple[str, EventRange]]]

    assert type_check(value, type)


@mark.parametrize("outer_type", (Sequence, List))
@mark.parametrize("inner_type value".split(), ( (int  , [1 ])
                                              , (str  , [""])
                                              , (float, [1.])
                                              , (dict , [{}])
                                              ))
def test_type_check_subscripted_generics(outer_type, inner_type, value):
    type = outer_type[inner_type]

    type_check(value, type)


def test_compare_signature_to_values_positional_only():
    class D(Enum):
        member = 0

    def f(a: int, b:float, c:str, d: D):
        return

    pos_values = (1, 2.0, "a_str", D.member)
    compare_signature_to_values(f, pos_values, {})


def test_compare_signature_to_values_keyword_only():
    class D(Enum):
        member = 0

    def f(a: int, b:float, c:str, d: D):
        return

    kwd_values = dict(a = 1, b = 2.0, c = "a_str", d = D.member)
    compare_signature_to_values(f, (), kwd_values)


def test_compare_signature_to_values_combined():
    class D(Enum):
        member = 0

    def f(a: int, b:float, c:str, d: D):
        return

    pos_values = (1, 2.0)
    kwd_values = dict(c = "a_str", d = D.member)
    compare_signature_to_values(f, pos_values, kwd_values)


def test_compare_signature_to_values_duck_match():
    def f(a: int, b:float):
        return

    values = dict(a = int(1), b = int(2))
    compare_signature_to_values(f, (), values)


@mark.parametrize("seq", (list, tuple, np.array))
def test_compare_signature_to_values_sequences(seq):
    def f(a: Sequence):
        return

    values = (seq([0, 1, 2]),)
    compare_signature_to_values(f, values, {})


def test_compare_signature_to_values_missing_without_default():
    def f(a: int, b:int, c:int):
        return

    pos_values = (1,)
    for arg_name in "bc":
        kwd_values = {arg_name: 1}
        match_str  = "The function .* is missing an argument .* of type .*"
        with raises(ValueError, match=match_str):
            compare_signature_to_values(f, pos_values, kwd_values)


@mark.parametrize("mode", "positional keyword".split())
def test_compare_signature_to_values_missing_with_default(mode):
    def f(a: int = 1, b:int = 2):
        return

    pos_values = (1,) if mode == "positional" else ()
    kwd_values = {}   if mode == "positional" else dict(b = 1)
    compare_signature_to_values(f, pos_values, kwd_values)


def test_compare_signature_to_values_unused_arguments():
    def f(a: int):
        return

    values    = dict(a=1, b=2)
    match_str = "Argument .* is not being used by .*"
    with warns(UserWarning, match=match_str):
        compare_signature_to_values(f, (), values)


@mark.parametrize("mode", "positional keyword".split())
@mark.parametrize("type1 value".split(), ( (int  ,       0)
                                         , (str  ,     "a")
                                         , (list ,     [1])
                                         , (dict , {2:"a"})
                                         , (tuple,    (3,))
                                         , (set  ,   {4,})))
@mark.parametrize("type2", (int, str, list, dict, tuple, set))
def test_compare_signature_to_values_raises(mode, type1, value, type2):
    if type1 is type2: return

    def f(a : type1):
        pass

    pos_values = (type2(),) if mode == "positional" else ()
    kwd_values = {}         if mode == "positional" else dict(a = type2())

    match_str = "The function .* expects an argument .* of type .*"
    with raises(ValueError, match=match_str):
        compare_signature_to_values(f, pos_values, kwd_values)


@mark.parametrize("do_check", (True, False))
def test_check_annotations(do_check):
    def f(a: int):
        return

    if do_check:
        # Type annotations are checked, raises an error
        f         = check_annotations(f)
        match_str = "The function .* expects an argument .* of type .*"
        with raises(ValueError, match=match_str):
            f(a = "the wrong type")
    else:
        # Type annotations not checked. Test passes
        f(a = "the wrong type")
