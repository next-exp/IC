"""Configure running options for the cities
JJGC August 2016
"""
import argparse
import sys
import os

from .        log_config      import logger
from . import system_of_units as     units


def print_configuration(options):
    """Print configuration.

    Parameters
    ----------
    options : namespace
        Contains attributes specifying the options
    """

    # Deal with namespcases as well as mappings.
    try:
        options = vars(options)
    except TypeError:
        pass

    for key, value in sorted(options.items()):
        print("{0: <22} => {1}".format(key, value))


def configure(input_options=sys.argv):
    """Translate command line options to a meaningful namespace.

    Parameters
    ----------
    input_options : sequence of strings, optional
        Input flags and parameters. Default are command line options from
        sys.argv.

    Returns
    -------
    output_options : namespace
        Options as attributes of a namespace.
    """
    program, *args = input_options
    parser = argparse.ArgumentParser(program)
    parser.add_argument("-c", metavar="cfile",     type=str, help="configuration file",             required=True)
    parser.add_argument("-i", metavar="ifile",     type=str, help="input file")
    parser.add_argument("-o", metavar="ofile",     type=str, help="output file")
    parser.add_argument("-n", metavar="nevt",      type=int, help="number of events to be processed")
    parser.add_argument("-f", metavar="firstevt",  type=int, help="event number for first event")
    parser.add_argument("-r", metavar="rnumber",   type=int, help="run number")
    parser.add_argument("-s", metavar="skip",      type=int, help="number of events to be skipped", default=0)
    parser.add_argument("-p", metavar="print_mod", type=int, help="print every this number of events")
    parser.add_argument("-I", action="store_true",           help="print info")
    parser.add_argument("-v", action="count",                help="verbosity level")
    parser.add_argument("--runall", action="store_true",     help="number of events to be skipped")

    flags, extras = parser.parse_known_args(args)
    options = read_config_file(flags.c) if flags.c else argparse.Namespace()

    if flags.i is not None: options.files_in     = flags.i
    if flags.r is not None: options.run_number   = flags.r
    if flags.o is not None: options.file_out     = flags.o
    if flags.n is not None: options.nevents      = flags.n
    if flags.f is not None: options.first_evt    = flags.f # TODO: do we still need this?
    if flags.s is not None: options.skip         = flags.s
    if flags.p is not None: options.print_mod    = flags.p
    if flags.v is not None: options.verbosity    = 50 - min(flags.v, 4) * 10
    if flags.runall:
        options.run_all = flags.runall
    options.info = flags.I

    if extras:
        logger.warning("WARNING: the following parameters have not been "
                       "identified!\n{}".format(" ".join(map(str, extras))))

    logger.setLevel(vars(options).get("verbosity", "info"))

    print_configuration(options)
    return options


def define_event_loop(options, n_evt):
    """Produce an iterator over the event numbers.

    Parameters
    ----------
    options : dictionary
        Contains the job parameters.
    n_evt : int
        Number of events in the input file.

    Returns
    ------
    gen : generator
        A generator producing the event numbers as configured in the job.
    """
    nevt = options.get("nevents", 0)
    max_evt = n_evt if options["run_all"] or nevt > n_evt else nevt
    start = options["skip"]
    print_mod = options.get("print_mod", max(1, (max_evt-start) // 20))

    for i in range(start, max_evt):
        if not i % print_mod:
            logger.info("Event # {}".format(i))
        yield i


def read_config_file(file_name):
    config = argparse.Namespace(verbosity=20)
    with open(file_name, 'r') as config_file:
        exec(config_file.read(), vars(units), vars(config))

    if hasattr(config, "path_in") and hasattr(config, "files_in"):
        config.files_in = os.path.expandvars(os.path.join(config.path_in, config.files_in))
        del config.path_in

    if hasattr(config, "path_out") and hasattr(config, "file_out"):
        config.file_out = os.path.expandvars(os.path.join(config.path_out, config.file_out))
        del config.path_out

    return config
