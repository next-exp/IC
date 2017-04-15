"""Configure running options for the cities
JJGC August 2016
"""
import argparse
import sys
import os

from invisible_cities.core.log_config import logger


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
    program, args = input_options[0], input_options[1:]
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

    if flags.i is not None: options.FILE_IN      = flags.i
    if flags.r is not None: options.RUN_NUMBER   = flags.r
    if flags.o is not None: options.FILE_OUT     = flags.o
    if flags.n is not None: options.NEVENTS      = flags.n
    if flags.f is not None: options.FIRST_EVT    = flags.f
    if flags.s is not None: options.SKIP         = flags.s
    if flags.p is not None: options.PRINT_MOD    = flags.p
    if flags.v is not None: options.VERBOSITY    = 50 - min(flags.v, 4) * 10
    if flags.runall:
        options.RUN_ALL = flags.runall
    options.INFO = flags.I

    if extras:
        logger.warning("WARNING: the following parameters have not been "
                       "identified!\n{}".format(" ".join(map(str, extras))))

    logger.setLevel(vars(options).get("VERBOSITY", "INFO"))

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
    nevt = options.get("NEVENTS", 0)
    max_evt = n_evt if options["RUN_ALL"] or nevt > n_evt else nevt
    start = options["SKIP"]
    print_mod = options.get("PRINT_MOD", max(1, (max_evt-start) // 20))

    for i in range(start, max_evt):
        if not i % print_mod:
            logger.info("Event # {}".format(i))
        yield i


def parse_value(value):
    """Parse booleans, ints on strings.

    Parameters
    ----------
    value : string
        Token to be converted.

    Returns
    -------
    converted_value : object
        Python object of the guessed type.
    """
    if value in ('True', 'False', 'None'): return eval(value)
    for parse in (int, float):
        try:                       return parse(value)
        except ValueError: pass
    else:                          return os.path.expandvars(value)


def read_config_file(cfile):
    """Read a configuration file of the form PARAMETER VALUE.

    Parameters
    ----------
    cfile : string
        Configuration file name (path included).

    Returns
    -------
    n : namespace
        Contains the parameters specified in cfile.
    """
    n = argparse.Namespace(VERBOSITY=20, RUN_ALL=False, COMPRESSION="ZLIB4")
    for line in open(cfile, "r"):
        line = line.split("#", 1)[0]

        if line.isspace() or line == "":
            continue

        # python-2 & python-3
        #tokens = [i for i in line.rstrip().split(" ") if i]
        # python-2 only. In python-2 filter returns a list in
        # python-3 filter retuns an iterator
        tokens = list(filter(None, line.rstrip().split(" ")))
        key = tokens[0]

        value = list(map(parse_value, tokens[1:]))  # python-2 & python-3
        vars(n)[key] = value[0] if len(value) == 1 else value

    if hasattr(n, "PATH_IN") and hasattr(n, "FILE_IN") and n.FILE_IN is not None:
        n.FILE_IN = os.path.join(n.PATH_IN, n.FILE_IN)
    if hasattr(n, "PATH_OUT") and hasattr(n, "FILE_OUT") and n.FILE_OUT is not None:
        n.FILE_OUT = os.path.join(n.PATH_OUT, n.FILE_OUT)
    return n


def filter_options(options, name):
    """Construct a new option dictionary with the parameters relevant to
    some module.

    Parameters
    ----------
    options : dictionary
        Dictionary of options with format "MODULE:PARAMETER": value.
    name : string
        Selected module name.

    Returns
    -------
    out : dictionary
        Filtered dictionary.

    """
    out = {}
    for key, value in options.items():
        if name in key:
            out[key.split(":")[1]] = value
    return out
