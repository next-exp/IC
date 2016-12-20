"""Table Functions
GML, October 2016

ChangeLog
14.10: import table io functions from wfmFunctions

19.10 copied functions that read tables from sensorFunctions. Keep the old
functions in sensorFunctions for now, give functions here more coherente names
(e.g, read_geom_table rather than read_data_geom). Function read_FEE_table
now returns also calibration constants for RWF and BLR (MC version)
"""

from __future__ import print_function

import numpy as np
import tables as tb
import pandas as pd

import Core.wfmFunctions as wfm
import Database.loadDB as DB
import Sierpe.FEE as FE


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
    if name == "NOCOMPR":
        return tb.Filters(complevel=0)  # no compression
    if name == "ZLIB1":
        return tb.Filters(complevel=1, complib="zlib")
    if name == "ZLIB4":
        return tb.Filters(complevel=4, complib="zlib")
    if name == "ZLIB5":
        return tb.Filters(complevel=5, complib="zlib")
    if name == "ZLIB9":
        return tb.Filters(complevel=9, complib="zlib")
    if name == "BLOSC5":
        return tb.Filters(complevel=5, complib="blosc")
    if name == "BLZ4HC5":
        return tb.Filters(complevel=5, complib="blosc:lz4hc")
    raise ValueError("Compression option {} not found.".format(name))


def store_FEE_table(fee_table):
    """Store the parameters of the EP FEE simulation."""
    DataPMT = DB.DataPMT()
    row = fee_table.row
    row["OFFSET"] = FE.OFFSET
    row["CEILING"] = FE.CEILING
    row["PMT_GAIN"] = FE.PMT_GAIN
    row["FEE_GAIN"] = FE.FEE_GAIN
    row["R1"] = FE.R1
    row["C1"] = FE.C1
    row["C2"] = FE.C2
    row["ZIN"] = FE.Zin
    row["DAQ_GAIN"] = FE.DAQ_GAIN
    row["NBITS"] = FE.NBITS
    row["LSB"] = FE.LSB
    row["NOISE_I"] = FE.NOISE_I
    row["NOISE_DAQ"] = FE.NOISE_DAQ
    row["t_sample"] = FE.t_sample
    row["f_sample"] = FE.f_sample
    row["f_mc"] = FE.f_mc
    row["f_LPF1"] = FE.f_LPF1
    row["f_LPF2"] = FE.f_LPF2
    row["coeff_c"] = DataPMT.coeff_c.values
    row["coeff_blr"] = DataPMT.coeff_blr.values
    row["adc_to_pes"] = DataPMT.adc_to_pes.values
    row["pmt_noise_rms"] = DataPMT.noise_rms
    row.append()
    fee_table.flush()


def read_FEE_table(fee_t):
    """Read the FEE table and return a PD Series for the simulation
    parameters and a PD series for the values of the capacitors used
    in the simulation.
    """

    fa = fee_t.read()

    F = pd.Series([fa[0][0], fa[0][1], fa[0][2], fa[0][3], fa[0][4],
                   fa[0][5], fa[0][6], fa[0][7], fa[0][8], fa[0][9],
                   fa[0][10], fa[0][11], fa[0][12], fa[0][13], fa[0][14],
                   fa[0][15], fa[0][16], fa[0][17]],
                  index=["OFFSET", "CEILING", "PMT_GAIN", "FEE_GAIN", "R1",
                         "C1", "C2", "ZIN", "DAQ_GAIN", "NBITS", "LSB",
                         "NOISE_I", "NOISE_DAQ", "t_sample", "f_sample",
                         "f_mc", "f_LPF1", "f_LPF2"])
    FEE = {}
    FEE["fee_param"] = F
    FEE["coeff_c"] = np.array(fa[0][18], dtype=np.double)
    FEE["coeff_blr"] = np.array(fa[0][19], dtype=np.double)
    FEE["adc_to_pes"] = np.array(fa[0][20], dtype=np.double)
    FEE["pmt_noise_rms"] = np.array(fa[0][21], dtype=np.double)
    return FEE


def store_deconv_table(table, params):
    row = table.row
    for param in table.colnames:
        row[param] = params[param]
    row.append()
    table.flush()


def read_deconv_table(table):
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
    pmttwf : tb.Table
        TWF table for PMTs
    sipmtwf : tb.Table
        TWF table for SiPMs
    pmtrwf : tb.EArray
        RWF array for PMTs
    pmtblr : tb.EArray
        BLR array for PMTs
    sipmrwf : tb.EArray
        RWF array for PMTs
    """
    pmttwf = h5f.root.TWF.PMT
    sipmtwf = h5f.root.TWF.SiPM
    pmtrwf = h5f.root.RD.pmtrwf
    pmtblr = h5f.root.RD.pmtblr
    sipmrwf = h5f.root.RD.sipmrwf
    return pmttwf, sipmtwf, pmtrwf, pmtblr, sipmrwf


def get_pmt_vectors(h5f):
    """Return the most relevant fields stored in a raw data file.

    Parameters
    ----------
    h5f : tb.File
        (Open) hdf5 file.

    Returns
    -------
    pmttwf : tb.Table
        TWF table for PMTs
    pmtrwf : tb.EArray
        RWF array for PMTs
    pmtblr : tb.EArray
        BLR array for PMTs
    """
    pmttwf = h5f.root.TWF.PMT
    pmtrwf = h5f.root.RD.pmtrwf
    pmtblr = h5f.root.RD.pmtblr
    return pmttwf, pmtrwf, pmtblr


def store_wf_table(event, table, wfdic, flush=True):
    """Store a set of waveforms in a table.

    Parameters
    ----------
    event : int
        Event number
    table : tb.Table
        Table instance where the wf must be stored.
    wfdic : dictionary or pd.Panel
        Contains a pd.DataFrame for each sensor. Keys are sensor IDs.
    flush : bool
        Whether to flush the table or not.
    """
    row = table.row
    for isens, wf in wfdic.iteritems():
        for t, e in zip(wf.time_mus, wf.ene_pes):
            row["event"] = event
            row["ID"] = isens
            row["time_mus"] = t
            row["ene_pes"] = e
            row.append()
    if flush:
        table.flush()


def read_sensor_wf(table, evt, isens):
    """Read back a particular waveform from a table.

    Parameters
    ----------
    table : tb.Table
        Table in which waveforms are stored.
    evt : int
        Event number
    isens : int
        Sensor number

    Returns
    -------
    time_mus : 1-dim np.ndarray
        Time samples of the waveform
    ene_pes : 1-dim np.ndarray
        Amplitudes of the waveform
    """
    return (table.read_where("(event=={}) & (ID=={})".format(evt, isens),
                             field="time_mus"),
            table.read_where("(event=={}) & (ID=={})".format(evt, isens),
                             field="ene_pes"))


def read_wf_table(table, event_number):
    """Read back a set of waveforms from a table.

    Parameters
    ----------
    table : tb.Table
        Table in which waveforms are stored.
    event_number : int
        Event number

    Returns
    -------
    wf_panel : pd.Panel
        pd.Panel with a pd.DataFrame for each sensor.
    """
    sensor_list = set(table.read_where("event == {}".format(event_number),
                      field="ID"))

    def get_df(isens):
        return wfm.wf2df(*read_sensor_wf(table, event_number, isens))

    return pd.Panel({isens: get_df(isens) for isens in sensor_list})


def store_pmap(pmap, table, evt, flush=True):
    """Store a pmap in a table.

    Parameters
    ----------
    pmap : Bridges.PMap
        PMap instance to be saved.
    table : tb.Table
        Table in which pmap will be stored.
    evt : int
        Event number
    flush : bool
        Whether to flush the table or not.
    """
    row = table.row
    for i, peak in enumerate(pmap.peaks):
        for time, ToT, e, qs in peak:
            row["event"] = evt
            row["peak"] = i
            row["signal"] = peak.signal
            row["time"] = time
            row["ToT"] = ToT
            row["cathode"] = e
            row["anode"] = qs
            row.append()
    if flush:
        table.flush()


def get_nofevents(table, column_name="evt_number"):
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
