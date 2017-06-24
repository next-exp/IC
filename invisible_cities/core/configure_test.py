from os import getenv, path
from pytest import mark

from . import configure as conf

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

# TODO remove! *ALL* possible parameters must now be set explicitly in
# the config files. (Users will be able to avoid having to specify
# them in their own config files by including the official config
# files for any city.)

# # Values that the configuration should assume if they are specified
# # neither in the config file, nor on the command line.
# default_config_spec = dict(run_all   = False,
#                            SKIP      =  0,
#                            VERBOSITY = 20)

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
                    (('nevents', '-n 99', 99),
                     ('skip'   , '-s 98', 98)),
                    # A long option in full
                    (('run_all' , '--runall', True),),
                    # A long option abbreviated
                    (('run_all' , '--ru', True),),
                    # Verbosity level 1
                    (('verbosity', '-v', 40),),
                    # Verbosity level 2
                    (('verbosity', '-v -v', 30),),
                    # Verbosity level 3
                    (('verbosity', '-v -v -v', 20),),
                    # Verbosity level 4
                    (('verbosity', '-v -v -v -v ', 10),),
                    # Verbosity level maxes out at 4
                    (('verbosity', '-v -v -v -v -v ', 10),),
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
    args_base = 'program_name -c {conf_file_name} '.format(**locals())
    args_options = ' '.join(extra_args)
    args = (args_base + args_options).split()

    # Present the command line to the parser
    CFP = conf.configure(args)

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
