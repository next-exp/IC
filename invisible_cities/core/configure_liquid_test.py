from os import path

from pytest import fixture
from pytest import mark
from pytest import raises

from . configure import all
from . configure import last
from . configure import configure
from . configure import Configuration
from . configure import read_config_file
from .           import system_of_units  as units

from . exceptions import NoInputFiles
from . exceptions import NoOutputFile

from .. liquid_cities.components  import city
from .. liquid_cities.penthesilea import penthesilea

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
config_file_spec = dict(files_in    = 'electrons_40keV_z250_RWF.h5',
                        file_out    = 'electrons_40keV_z250_PMP.h5',
                        compression = 'ZLIB4',
                        run_number  = 23,
                        nprint      = 24,
                        nbaseline   = 26,
                        thr_trigger = 27,
                        nmau        = 28,
                        thr_mau     = 29,
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
        conf = configure(argv)


def test_configure_raises_SystemExit_with_multiple_mutually_exclusive_options():
    argv = f"dummy {default_conf} --no-files --full-files".split()
    with raises(SystemExit):
        conf = configure(argv)


def test_read_config_file_special_values(config_tmpdir):
    filename  = path.join(config_tmpdir, "test_read_config_file_special_values.conf")
    all_units = list(vars(units).keys())
    write_config_file(filename, f"""
var_all    = all
var_last   = last
vars_units = {all_units}
""")
    argv = f"dummy {default_conf} -i ifile -o ofile -r runno".split()
    conf = read_config_file(filename)
    assert conf["var_all"   ] is all
    assert conf["var_last"  ] is last
    assert conf["vars_units"] == all_units


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


@mark.parametrize('flags value counter'.split(),
                  (('-e all'   , 10, 'events_in'), # 10 events in the file
                   ('-e   9'   ,  9, 'events_in'), # [ 0,  9) -> 9
                   ('-e 5 9'   ,  4, 'events_in'), # [ 5,  9) -> 4
                   ('-e 2 last',  8, 'events_in'), # events [2, 10) -> 8
                  ))
def test_config_penthesilea_counters(config_tmpdir, KrMC_pmaps_filename, flags, value, counter):
    input_filename  = KrMC_pmaps_filename
    config_filename = 'invisible_cities/config/liquid_penthesilea.conf'
    flags_wo_spaces = flags.replace(" ", "_")
    output_filename = path.join(config_tmpdir,
                                f'penthesilea_counters_output_{flags_wo_spaces}.h5')
    argv = f'penthesilea {config_filename} -i {input_filename} -o {output_filename} {flags}'.split()
    counters = penthesilea(**configure(argv))
    assert getattr(counters, counter) == value
