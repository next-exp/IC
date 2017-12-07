from os import getenv
from os import path

import pytest
from   pytest import mark
from   pytest import raises

from . configure import all
from . configure import last
from . configure import configure
from . configure import Configuration
from . configure import make_config_file_reader

from . exceptions import NoInputFiles
from . exceptions import NoOutputFile

from .. cities.base_cities import City
from .. cities.penthesilea import Penthesilea


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

# set_mau
nmau    = {nmau}
thr_mau = {thr_mau}

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
config_file_spec = dict(files_in = 'electrons_40keV_z250_RWF.h5',
                        file_out = 'electrons_40keV_z250_PMP.h5',
                        compression        = 'ZLIB4',
                        run_number         = 23,
                        nprint             = 24,
                        nbaseline          = 26,
                        thr_trigger        = 27,
                        nmau               = 28,
                        thr_mau            = 29,
                        thr_csum           =  0.5,
                        s1_tmin            = 31,
                        s1_tmax            = 32,
                        s1_stride          = 33,
                        s1_lmin            = 34,
                        s1_lmax            = 35,
                        s2_tmin            = 36,
                        s2_tmax            = 37,
                        s2_stride          = 39,
                        s2_lmin            = 41,
                        s2_lmax            = 42,
                        thr_zs             = 43,
                        thr_sipm_s2        = 44,
                        event_range        = (45, 46))

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
def test_configure(config_tmpdir, spec):

    conf_file_name = config_tmpdir.join('test.conf')
    with open(conf_file_name, 'w') as conf_file:
        conf_file.write(config_file_contents)

    # Extract command line aruments and the corresponding desired
    # values from the test's parameter.
    extra_args   = [    arg       for (_, arg, _    ) in spec ]
    cmdline_spec = { opt : value  for (opt, _, value) in spec }

    # Compose the command line
    args_base = 'program_name {conf_file_name} '.format(**locals())
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


def write_config_file(file_name, contents):
    with open(file_name, 'w') as out:
        out.write(contents)


@pytest.fixture(scope = 'session')
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


@pytest.fixture(scope = 'session')
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


class DummyCity(City):

    def __init__(self, **kwds):
        super().__init__(**kwds)
        self.cnt.n_events_tot = 10

    def file_loop(self): pass
    def get_writers(self, h5out): pass
    def write_parameters(self, h5out): pass


def test_config_drive_fails_without_config_file():
    argv = 'dummy'.split()
    with raises(SystemExit):
        DummyCity.drive(argv)

def test_config_drive_fails_without_input_file(simple_conf_file_name):
    argv = 'dummy {simple_conf_file_name}'.format(**locals()).split()
    with raises(NoInputFiles):
        DummyCity.drive(argv)

def test_config_drive_fails_without_output_file(simple_conf_file_name):
    conf   = simple_conf_file_name
    infile = conf # any existing file will do as a dummy for now
    argv = 'dummy {conf} -i {infile}'.format(**locals()).split()
    with raises(NoOutputFile):
        DummyCity.drive(argv)


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
def test_config_drive_flags(simple_conf_file_name, tmpdir_factory, name, flags, value):
    conf   = simple_conf_file_name
    infile = conf # any existing file will do as a dummy for now
    outfile = tmpdir_factory.mktemp('drive-config').join('dummy-output-file' + name)
    argv = 'dummy {conf} -i {infile} -o {outfile} {flags}'.format(**locals()).split()
    conf_ns, _ = DummyCity.drive(argv)
    assert getattr(conf_ns, name) == value


@mark.parametrize('flags value counter'.split(),
                  (('-e all'   , 10, 'n_events_tot'), # 10 events in the file
                   ('-e   9'   ,  9, 'n_events_tot'), # [ 0,  9) -> 9
                   ('-e 5 9'   ,  4, 'n_events_tot'), # [ 5,  9) -> 4
                   ('-e 2 last',  8, 'n_events_tot'), # events [2, 10) -> 8
                  ))
def test_config_drive_penthesilea_counters(config_tmpdir, KrMC_pmaps_filename, flags, value, counter):
    input_filename  = KrMC_pmaps_filename
    config_filename = 'invisible_cities/config/penthesilea.conf'
    flags_wo_spaces = flags.replace(" ", "_")
    output_filename = path.join(config_tmpdir,
                                f'penthesilea_counters_output_{flags_wo_spaces}.h5')
    argv = f'penthesilea {config_filename} -i {input_filename} -o {output_filename} {flags}'.split()
    conf_ns, counters = Penthesilea.drive(argv)
    assert getattr(counters, counter) == value
