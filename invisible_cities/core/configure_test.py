from os import getenv, path
from pytest import mark
from hypothesis import given, example
from hypothesis.strategies import integers, floats, one_of, none

from . import configure as conf

config_file_format = """
# set_input_files
PATH_IN {PATH_IN} # comment to be ignored 1
FILE_IN {FILE_IN}

# set_pmap_store
PATH_OUT {PATH_OUT}
FILE_OUT {FILE_OUT}
COMPRESSION {COMPRESSION}

# irene
RUN_NUMBER {RUN_NUMBER}

# set_print
NPRINT {NPRINT}

# print empty events (skipped)
PRINT_EMPTY_EVENTS {PRINT_EMPTY_EVENTS}

# set_blr
NBASELINE {NBASELINE}
THR_TRIGGER {THR_TRIGGER}

# set_mau
NMAU {NMAU}
THR_MAU {THR_MAU}

# set_csum
THR_CSUM {THR_CSUM}

# set_s1
S1_TMIN {S1_TMIN}
S1_TMAX {S1_TMAX}
S1_STRIDE {S1_STRIDE}
S1_LMIN {S1_LMIN}
S1_LMAX {S1_LMAX}

# set_s2
S2_TMIN {S2_TMIN}
S2_TMAX {S2_TMAX}
S2_STRIDE {S2_STRIDE}
S2_LMIN {S2_LMIN}
S2_LMAX {S2_LMAX}

# set_sipm
THR_ZS {THR_ZS}
THR_SIPM_S2 {THR_SIPM_S2}

# run
NEVENTS {NEVENTS}
RUN_ALL {RUN_ALL}
"""

# The values that will be fed into the above.
config_file_spec = dict(PATH_IN  = '$ICDIR/database/test_data/',
                        FILE_IN  = 'electrons_40keV_z250_RWF.h5',
                        PATH_OUT = '$ICDIR/database/test_data/',
                        FILE_OUT = 'electrons_40keV_z250_PMP.h5',
                        COMPRESSION        = None,
                        RUN_NUMBER         = 23,
                        NPRINT             = 24,
                        PRINT_EMPTY_EVENTS = 25,
                        NBASELINE          = 26,
                        THR_TRIGGER        = 27,
                        NMAU               = 28,
                        THR_MAU            = 29,
                        THR_CSUM           =  0.5,
                        S1_TMIN            = 31,
                        S1_TMAX            = 32,
                        S1_STRIDE          = 33,
                        S1_LMIN            = 34,
                        S1_LMAX            = 35,
                        S2_TMIN            = 36,
                        S2_TMAX            = 37,
                        S2_STRIDE          = 39,
                        S2_LMIN            = 41,
                        S2_LMAX            = 42,
                        THR_ZS             = 43,
                        THR_SIPM_S2        = 44,
                        RUN_ALL            = False,
                        NEVENTS            = 45)

# Values that the configuration should assume if they are specified
# neither in the config file, nor on the command line.
default_config_spec = dict(INFO      = False,
                           RUN_ALL   = False,
                           SKIP      =  0,
                           VERBOSITY = 20)

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
                    (('NEVENTS', '-n 99', 99),
                     ('SKIP'   , '-s 98', 98)),
                    # A long option in full
                    (('RUN_ALL' , '--runall', True),),
                    # A long option abbreviated
                    (('RUN_ALL' , '--ru', True),),
                    # Verbosity level 1
                    (('VERBOSITY', '-v', 40),),
                    # Verbosity level 2
                    (('VERBOSITY', '-v -v', 30),),
                    # Verbosity level 3
                    (('VERBOSITY', '-v -v -v', 20),),
                    # Verbosity level 4
                    (('VERBOSITY', '-v -v -v -v ', 10),),
                    # Verbosity level maxes out at 4
                    (('VERBOSITY', '-v -v -v -v -v ', 10),),
                    # Input and output files
                    (('FILE_IN',  '-i some_input_file',  'some_input_file'),
                     ('FILE_OUT', '-o some_output_file', 'some_output_file')),
                    # Info on
                    (('INFO', '-I', True),),
                    # Info on
                    (('INFO', '',   False),),
                  ))
def test_configure(config_tmpdir, spec):
    """Test configure function. Read from conf file.
    """

    conf_file_name = str(config_tmpdir.join('test.conf'))
    with open(conf_file_name, 'w') as conf_file:
        conf_file.write(config_file_contents)

    # Extract command line aruments and the corresponding desired
    # values from the test's parameter.
    extra_args   = [    arg       for (_, arg, _    ) in spec ]
    cmdline_spec = { opt : value  for (opt, _, value) in spec }

    # Compose the command line
    args_base = 'program_name -c {conf_file_name} '.format(**locals())
    args_options = ' '.join(extra_args)
    args = (args_base + args_options).split()

    # Present the command line to the parser
    CFP = conf.configure(args)

    # The values in the configuration can come from three different places
    # 1. Default values
    # 2. The configuration file can override these
    # 3. Command line arguments can override all of these
    this_configuration = join_dicts(default_config_spec,
                                    config_file_spec,
                                    cmdline_spec)

    # Expand environment variables in paths
    for k,v in this_configuration.items():
        try:
            if '$' in v:
                this_configuration[k] = path.expandvars(v)
        except TypeError:
            pass

    # FILE_IN and FILE_OUT have to be treated specially: the
    # corresponding PATHs must be prepended, unless the file is
    # specified on the command line.
    if not ('FILE_IN' in cmdline_spec):
        this_configuration['FILE_IN']  = path.join(this_configuration['PATH_IN'],
                                                   this_configuration['FILE_IN'])
    if not ('FILE_OUT' in cmdline_spec):
        this_configuration['FILE_OUT'] = path.join(this_configuration['PATH_OUT'],
                                                   this_configuration['FILE_OUT'])

    # Check that all options have the expected values
    for option in this_configuration:
        assert getattr(CFP, option) == this_configuration[option], 'option = ' + option



@given(one_of(integers(),
              floats(allow_nan=False, allow_infinity=False)),
       none())
@example(True,  True)
@example(False, False)
@example(None, None)
@example(    '$PWD',          getenv('PWD'))
@example('xxx/$PWD', 'xxx/' + getenv('PWD'))
@example('astring', 'astring')
@example('spa ace', 'spa ace')
def test_parse_value(i,o):
    # `given` generates integers or floats as input: their string
    # representation needs to be seen by `parse_value`.
    input  = str(i)
    # `given` generates `None` as the correct answer, signalling to
    # the test that the answer should be obtained by evaluating the
    # input; `example` passes in the exact correct answer, signalling
    # that that is the value we should use in the assertion.
    output = eval(input) if o is None else o

    assert conf.parse_value(input) == output
