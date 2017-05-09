"""Table Functions
GML, October 2016

ChangeLog
14.10: import table io functions from wfmFunctions

19.10 copied functions that read tables from sensorFunctions. Keep the old
functions in sensorFunctions for now, give functions here more coherente names
(e.g, read_geom_table rather than read_data_geom). Function read_FEE_table
now returns also calibration constants for RWF and BLR (MC version)
"""

import numpy as np
import tables as tb
import pandas as pd
from argparse import Namespace

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


def read_FEE_table(fee_t):
    """Read the FEE table and return a PD Series for the simulation
    parameters and a PD series for the values of the capacitors used
    in the simulation.
    """

    fa = fee_t.read()

    F = pd.Series([fa[0][ 0], fa[0][ 1], fa[0][ 2], fa[0][ 3], fa[0][ 4],
                   fa[0][ 5], fa[0][ 6], fa[0][ 7], fa[0][ 8], fa[0][ 9],
                   fa[0][10], fa[0][11], fa[0][12], fa[0][13], fa[0][14],
                   fa[0][15], fa[0][16], fa[0][17]],
                  index=["OFFSET", "CEILING", "PMT_GAIN", "FEE_GAIN", "R1",
                         "C1", "C2", "ZIN", "DAQ_GAIN", "NBITS", "LSB",
                         "NOISE_I", "NOISE_DAQ", "t_sample", "f_sample",
                         "f_mc", "f_LPF1", "f_LPF2"])
    FEE = Namespace()
    FEE.fee_param     = F
    FEE.coeff_c       = np.array(fa[0][18], dtype=np.double)
    FEE.coeff_blr     = np.array(fa[0][19], dtype=np.double)
    FEE.adc_to_pes    = np.array(fa[0][20], dtype=np.double)
    FEE.pmt_noise_rms = np.array(fa[0][21], dtype=np.double)
    return FEE


def table_from_params(table, params):
    row = table.row
    for param in table.colnames:
        row[param] = params[param]
    row.append()
    table.flush()


def table_to_params(table):
    params = {}
    for param in table.colnames:
        params[param] = table[0][param]
    return params


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

    pmtrwf = h5f.root.RD.pmtrwf
    pmtblr = h5f.root.RD.pmtblr
    sipmrwf = h5f.root.RD.sipmrwf
    return pmtrwf, pmtblr, sipmrwf


def get_nof_events(table, column_name="evt_number"):
    """Find number of events in table by asking number of different values
    in column.

    Parameters
    ----------
    table : tb.Table
        Table to be read.
    column_name : string
        Name of the column with a unique value for each event.

    Returns
    -------
    nevt : int
        Number of events in table.

    """
    return len(set(table.read(field=column_name)))


def get_event_numbers_and_timestamps(filename):
    with tb.open_file(filename, "r") as h5in:
        event_numbers = h5in.root.Run.events.cols.evt_number[:]
        timestamps    = h5in.root.Run.events.cols.timestamp [:]
        return event_numbers, timestamps