"""Configure running options for the cities
JJGC August 2016
"""
import argparse
import sys
import os

from . log_config import logger


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
    if value in ('True', 'False'): return eval(value)
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
    n = argparse.Namespace(verbosity=20, run_all=False, compression="ZLIB4")
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

    if hasattr(n, "path_in") and hasattr(n, "files_in"):
        n.files_in = os.path.join(n.path_in, n.files_in)
        del n.path_in

    if hasattr(n, "path_out") and hasattr(n, "file_out"):
        n.file_out = os.path.join(n.path_out, n.file_out)
        del n.path_out
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
