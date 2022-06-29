"""Table Functions
GML, October 2016

ChangeLog
14.10: import table io functions from wfmFunctions

19.10 copied functions that read tables from sensorFunctions. Keep the old
functions in sensorFunctions for now, give functions here more coherente names
(e.g, read_geom_table rather than read_data_geom). Function read_FEE_table
now returns also calibration constants for RWF and BLR (MC version)
"""

import re
import numpy as np
import tables as tb


def filters(name):
    """Return the filter corresponding to a given key.

    Parameters
    ----------
    name : string
        Label of the compression mode and level. Options are:
        - NOCOMPR: no compression
        - ZLIB(1,4,5,9): ZLIB library with compression level (1,4,5,9)
        - BLOSC(5): BLOSC library with compresion level 5
        - BLZ4HC(5): BLOSC library with codec lz4hc and compression level 5

    Returns
    -------
    filt : tb.filters.Filter
        Filter mode instance.
    """
    if name is None: return None

    try:
        level, lib = {"NOCOMPR": (0,  None)        ,
                      "ZLIB1"  : (1, 'zlib')       ,
                      "ZLIB4"  : (4, 'zlib')       ,
                      "ZLIB5"  : (5, 'zlib')       ,
                      "ZLIB9"  : (9, 'zlib')       ,
                      "BLOSC5" : (5, 'blosc')      ,
                      "BLZ4HC5": (5, 'blosc:lz4hc'),
                      }[name]
        return tb.Filters(complevel=level, complib=lib)
    except KeyError:
        raise ValueError("Compression option {} not found.".format(name))


def get_vectors(h5f):
    """Return the most relevant fields stored in a raw data file.

    Parameters
    ----------
    h5f : tb.File
        (Open) hdf5 file.

    Returns
    -------

    pmtrwf : tb.EArray
        RWF array for PMTs
    pmtblr : tb.EArray
        BLR array for PMTs
    sipmrwf : tb.EArray
        RWF array for PMTs
    """

    pmtrwf = np.zeros((1, 12, 1))
    pmtblr = np.zeros((1, 12, 1))
    sipmrwf = np.zeros((1, 1792, 1))
    if 'pmtrwf' in h5f.root.RD:
        pmtrwf = h5f.root.RD.pmtrwf

    if 'pmtblr' in h5f.root.RD:
        pmtblr = h5f.root.RD.pmtblr

    if 'sipmrwf' in h5f.root.RD:
        sipmrwf = h5f.root.RD.sipmrwf

    return pmtrwf, pmtblr, sipmrwf


def event_number_from_input_file_name(filename):
    # We use a regular expression to get the file number, this is the meaning:
    # NEXT_v\d[_\d+]+   -> Get software version, example: NEXT_v0_08_09
    # [_\w]+?           -> Matches blocks with _[a-zA-Z0-9] in a non-greedy way
    # _(?P<number>\d+)_ -> Matches the file number and save it with name 'fnumber'
    # [_\w]+?           -> Matches blocks with _[a-zA-Z0-9] in a non-greedy way
    # (?P<nevts>\d+)    -> Matches number of events and save it with name 'nevts'
    # \..*              -> Matches a dot and the rest of the name
    # Sample name: 'dst_NEXT_v0_08_09_Co56_INTERNALPORTANODE_74_0_7bar_MCRD_10000.root.h5'
    pattern = re.compile(r'NEXT_v\d[_\d+]+[_\w]+?_(?P<fnumber>\d+)_[_\w]+?_(?P<nevts>\d+)\..*',
                         re.IGNORECASE)
    match = pattern.search(filename)
    # If string does not match the pattern, return 0 as default
    filenumber = 0
    nevts      = 0
    if match:
        filenumber = int(match.group('fnumber'))
        nevts      = int(match.group('nevts'))
    return filenumber * nevts
