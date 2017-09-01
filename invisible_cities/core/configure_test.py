from os import getenv, path

import pytest
from   pytest import mark
from   pytest import raises

from . configure import configure
from . configure import Configuration
from . configure import make_config_file_reader

from . exceptions import NoInputFiles
from . exceptions import NoOutputFile

from .. cities.base_cities import City


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

# print empty events (skipped)
print_empty_events = {print_empty_events}

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
nevents = {nevents}
run_all = {run_all}
"""

# The values that will be fed into the above.
config_file_spec = dict(files_in = 'electrons_40keV_z250_RWF.h5',
                        file_out = 'electrons_40keV_z250_PMP.h5',
                        compression        = 'ZLIB4',
                        run_number         = 23,
                        nprint             = 24,
                        print_empty_events = 25,
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
                        run_all            = False,
                        nevents            = 45)

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
                    (('nevents'    , '-n 99', 99),
                     ('first_event', '-f 98', 98)),
                    # A long option in full
                    (('run_all' , '--run-all', True),),
                    # A long option abbreviated
                    (('nevents' , '--ne 6', 6),),
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



@pytest.fixture(scope = 'session')
def simple_conf_file_name(tmpdir_factory):
    dir_ = tmpdir_factory.mktemp('test_config_files')
    file_name = path.join(dir_, 'simple.conf')
    write_config_file(file_name, """
compression = 'ZLIB4'
run_number  = 12
nprint      = 13
nmax        = 14
""")
    return str(file_name)


class DummyCity(City):

    def __init__(self, **kwds):
        super().__init__(**kwds)
        self.cnt.init_counter('n_events_tot', value=10)

    def file_loop(self): pass


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

def test_config_drive_flags(simple_conf_file_name, tmpdir_factory):
    conf   = simple_conf_file_name
    infile = conf # any existing file will do as a dummy for now
    outfile = tmpdir_factory.mktemp('drive-config').join('dummy-output-file-s')
    argv = 'dummy {conf} -i {infile} -o {outfile}'.format(**locals()).split()
    conf = DummyCity.drive(argv)
    assert conf.first_event == 0

@mark.parametrize(  'name          flags            value'.split(),
                  (('first_event',              '-f 21', 21),
                   ('first_event',   '--first-event 22', 22),
                   ('run_number',               '-r 23', 23),
                   ('run_number',     '--run-number 24', 24),
                  ))
def test_config_drive_first_event_short(simple_conf_file_name, tmpdir_factory, name, flags, value):
    conf   = simple_conf_file_name
    infile = conf # any existing file will do as a dummy for now
    outfile = tmpdir_factory.mktemp('drive-config').join('dummy-output-file' + name)
    argv = 'dummy {conf} -i {infile} -o {outfile} {flags}'.format(**locals()).split()
    print(argv)
    conf = DummyCity.drive(argv)
    assert conf.first_event == value
