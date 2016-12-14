"""
Configure running options for the cities
JJGC August 2016
"""
import argparse
import sys
import os

from Core.LogConfig import logger


def print_configuration(options):
    """
    Print configuration.

    Parameters
    ----------
    options : dictionary
        Contains key-value pairs of options.
    """

    for key, value in sorted(options.items()):
        print("{0: <22} => {1}".format(key, value))


def configure(input_options=sys.argv):
    """
    Translate command line options to a meaningfull dictionary.

    Parameters
    ----------
    input_options : sequence of strings, optional
        Input flags and parameters. Default are command line options from
        sys.argv.

    Returns
    -------
    output_options : dictionary
        Options as key-value pairs.
    """
    program, args = input_options[0], input_options[1:]
    parser = argparse.ArgumentParser(program)
    parser.add_argument("-c", metavar="cfile", type=str,
                        help="configuration file", required=True)
    parser.add_argument("-i", metavar="ifile", type=str,
                        help="input file")
    parser.add_argument("-o", metavar="ofile", type=str,
                        help="output file")
    parser.add_argument("-n", metavar="nevt", type=int,
                        help="number of events to be processed")
    parser.add_argument("-s", metavar="skip", type=int, default=0,
                        help="number of events to be skipped")
    parser.add_argument("-p", metavar="print_mod", type=int,
                        help="print every this number of events")
    parser.add_argument("--runall", action="store_true",
                        help="number of events to be skipped")
    parser.add_argument("-I", action="store_true", help="print info")
    parser.add_argument("-v", action="count", help="verbosity level")

    flags, extras = parser.parse_known_args(args)
    options = read_config_file(flags.c) if flags.c else {}

    if flags.i is not None:
        options["FILE_IN"] = flags.i
    if flags.o is not None:
        options["FILE_OUT"] = flags.o
    if flags.n is not None:
        options["NEVENTS"] = flags.n
    if flags.s is not None:
        options["SKIP"] = flags.s
    if flags.p is not None:
        options["PRINT_MOD"] = flags.p
    if flags.runall:
        options["RUN_ALL"] = flags.runall
    options["INFO"] = flags.I
    if flags.v is not None:
        options["VERBOSITY"] = 50 - min(flags.v, 4)*10

    if extras:
        logger.warning("WARNING: the following parameters have not been "
                       "identified!\n{}".format(" ".join(map(str, extras))))

    logger.setLevel(options.get("VERBOSITY", "INFO"))

    print_configuration(options)
    return options


def define_event_loop(options, n_evt):
    """
    Produce an iterator over the event numbers.

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
    print_mod = options.get("PRINT_MOD", max(1, (max_evt-start)//20))

    for i in range(start, max_evt):
        if not i % print_mod:
            logger.info("Event # {}".format(i))
        yield i


def cast(value):
    """
    Cast value from string to a python type.

    Parameters
    ----------
    value : string
        Token to be casted.

    Returns
    -------
    casted_value : variable
        Python variable of the guessed type.
    """
    if value == "True":
        return True
    if value == "False":
        return False
    if value.isdigit():
        return int(value)
    if value.replace(".", "").isdigit():
        return float(value)
    if "$" in value:
        value = os.path.expandvars(value)
    return value


def read_config_file(cfile):
    """
    Read a configuration file of the form PARAMETER VALUE.

    Parameters
    ----------
    cfile : string
        Configuration file name (path included).

    Returns
    -------
    d : dictionary
        Contains the parameters specified in cfile.
    """
    d = {"VERBOSITY": 20, "RUN_ALL": False, "COMPRESSION": "ZLIB4"}
    for line in open(cfile, "r"):
        if line == "\n" or line[0] == "#":
            continue
        tokens = filter(lambda x: x != "", line.rstrip().split(" "))
        key = tokens[0]

        value = map(cast, tokens[1:])
        d[key] = value[0] if len(value) == 1 else value

    if "PATH_IN" in d and "FILE_IN" in d:
        d["FILE_IN"] = d["PATH_IN"] + "/" + d["FILE_IN"]
        del d["PATH_IN"]
    if "PATH_OUT" in d and "FILE_OUT" in d:
        d["FILE_OUT"] = d["PATH_OUT"] + "/" + d["FILE_OUT"]
        del d["PATH_OUT"]
    return d


def filter_options(options, name):
    """
    Construct a new option dictionary with the parameters relevant to some
    module.

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
