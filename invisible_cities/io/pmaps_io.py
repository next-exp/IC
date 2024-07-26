from functools import partial

import numpy  as np
import tables as tb
import pandas as pd

from .. evm .pmaps         import  PMTResponses
from .. evm .pmaps         import SiPMResponses
from .. evm .pmaps         import S1
from .. evm .pmaps         import S2
from .. evm .pmaps         import PMap
from .. evm                import nh5     as table_formats
from .. core.tbl_functions import filters as tbl_filters


def store_peak(pmt_table, pmti_table, si_table,
               peak, peak_number, event_number):
    pmt_row  =  pmt_table.row
    pmti_row = pmti_table.row

    for i, t in enumerate(peak.times):
        pmt_row['event' ] = event_number
        pmt_row['peak'  ] = peak_number
        pmt_row['time'  ] = t
        pmt_row['bwidth'] = peak.bin_widths[i]
        pmt_row['ene'   ] = peak.pmts.sum_over_sensors[i]
        pmt_row.append()

    for pmt_id in peak.pmts.ids:
        for e in peak.pmts.waveform(pmt_id):
            pmti_row['event'] = event_number
            pmti_row['peak' ] =  peak_number
            pmti_row['npmt' ] = pmt_id
            pmti_row['ene'  ] = e
            pmti_row.append()

    if si_table is None: return

    si_row = si_table.row
    for sipm_id in peak.sipms.ids:
        for q in peak.sipms.waveform(sipm_id):
            si_row['event'] = event_number
            si_row['peak' ] =  peak_number
            si_row['nsipm'] = sipm_id
            si_row['ene'  ] = q
            si_row.append()


def store_pmap(tables, pmap, event_number):
    s1_table, s2_table, si_table, s1i_table, s2i_table = tables
    for peak_number, s1 in enumerate(pmap.s1s):
        store_peak(s1_table, s1i_table,     None, s1, peak_number, event_number)
    for peak_number, s2 in enumerate(pmap.s2s):
        store_peak(s2_table, s2i_table, si_table, s2, peak_number, event_number)


def pmap_writer(file, *, compression=None):
    tables = _make_tables(file, compression)
    return partial(store_pmap, tables)


def _make_tables(hdf5_file, compression):
    compr       = tbl_filters(compression)
    pmaps_group = hdf5_file.create_group(hdf5_file.root, 'PMAPS')
    make_table  = partial(hdf5_file.create_table, pmaps_group, filters=compr)

    s1    = make_table('S1'   , table_formats.S12   ,    "S1 Table")
    s2    = make_table('S2'   , table_formats.S12   ,    "S2 Table")
    s2si  = make_table('S2Si' , table_formats.S2Si  ,  "S2Si Table")
    s1pmt = make_table('S1Pmt', table_formats.S12Pmt, "S1Pmt Table")
    s2pmt = make_table('S2Pmt', table_formats.S12Pmt, "S2Pmt Table")

    pmp_tables = s1, s2, s2si, s1pmt, s2pmt
    for table in pmp_tables:
        # Mark column to be indexed
        table.set_attr('columns_to_index', ['event'])

    return pmp_tables


def load_pmaps_as_df(filename):
    with tb.open_file(filename, 'r') as h5f:
        pmap  = h5f.root.PMAPS
        to_df = pd.DataFrame.from_records
        return (to_df(pmap.S1   .read()),
                to_df(pmap.S2   .read()),
                to_df(pmap.S2Si .read()),
                to_df(pmap.S1Pmt.read()) if 'S1Pmt' in pmap else None,
                to_df(pmap.S2Pmt.read()) if 'S2Pmt' in pmap else None)


# Hack fix to allow loading pmaps without individual pmts. Used in load_pmaps
def _build_ipmtdf_from_sumdf(sumdf):
    ipmtdf = sumdf.copy()
    ipmtdf = ipmtdf.rename(index=str, columns={'time': 'npmt'})
    ipmtdf['npmt'] = -1
    return ipmtdf

def load_pmaps(filename):
    pmap_dict = {}
    s1df, s2df, sidf, s1pmtdf, s2pmtdf = load_pmaps_as_df(filename)
    # Hack fix to allow loading pmaps without individual pmts
    if s1pmtdf is None: s1pmtdf = _build_ipmtdf_from_sumdf(s1df)
    if s2pmtdf is None: s2pmtdf = _build_ipmtdf_from_sumdf(s2df)
    event_numbers = set.union(set(s1df.event), set(s2df.event))
    s1df_grouped   =s1df   .groupby('event')
    s1pmtdf_grouped=s1pmtdf.groupby('event')
    s2df_grouped   =s2df   .groupby('event')
    s2pmtdf_grouped=s2pmtdf.groupby('event')
    sidf_grouped   =sidf   .groupby('event')
    for event_number in event_numbers:
        try:
            s1indx=s1df_grouped      .groups[event_number]
        except KeyError:
            s1indx=[]
        try:
            s1pmtindx=s1pmtdf_grouped.groups[event_number]
        except KeyError:
            s1pmtindx=[]
        try:
            s2indx=s2df_grouped      .groups[event_number]
        except KeyError:
            s2indx=[]
        try:
            s2pmtindx=s2pmtdf_grouped.groups[event_number]
        except KeyError:
            s2pmtindx=[]
        try:
            siindx=sidf_grouped      .groups[event_number]
        except KeyError:
            siindx=[]

        s1s = s1s_from_df(s1df   .iloc   [s1indx   ],
                          s1pmtdf.iloc   [s1pmtindx])
        s2s = s2s_from_df(s2df   .iloc   [s2indx   ],
                          s2pmtdf.iloc   [s2pmtindx],
                          sidf   .iloc   [siindx   ])

        pmap_dict[event_number] = PMap(s1s, s2s)

    return pmap_dict


def build_pmt_responses(pmtdf, ipmtdf):
    times = pmtdf.time.values
    try:
        widths = pmtdf.bwidth.values
    except AttributeError:
        ## Old file without bin widths saved
        ## Calculate 'fake' widths from times
        time_diff = np.diff(times)
        if len(time_diff) == 0:
            widths = np.full(1, 1000)
        elif np.all(time_diff == time_diff[0]):
            ## S1-like
            widths = np.full(times.shape, time_diff[0])
        else:
            ## S2-like, round to closest mus
            binw = time_diff.max().round(-3)
            widths = np.full(times.shape, binw)
    pmt_ids = pd.unique(ipmtdf.npmt.values)
    enes    =           ipmtdf.ene .values.reshape(pmt_ids.size,
                                                     times.size)
    return times, widths, PMTResponses(pmt_ids, enes)


def build_sipm_responses(sidf):
    if len(sidf) == 0: return SiPMResponses.build_empty_instance()

    sipm_ids = pd.unique(sidf.nsipm.values)
    enes     =           sidf.ene  .values
    n_times  = enes.size // sipm_ids.size
    enes     = enes.reshape(sipm_ids.size, n_times)
    return SiPMResponses(sipm_ids, enes)


def s1s_from_df(s1df, s1pmtdf):
    s1s = []
    peak_numbers = set(s1df.peak)
    for peak_number in peak_numbers:
        (times ,
         widths,
         pmt_r ) = build_pmt_responses(s1df   [s1df   .peak == peak_number],
                                       s1pmtdf[s1pmtdf.peak == peak_number])
        s1s.append(S1(times, widths,
                      pmt_r, SiPMResponses.build_empty_instance()))

    return s1s


def s2s_from_df(s2df, s2pmtdf, sidf):
    s2s = []
    peak_numbers = set(s2df.peak)
    for peak_number in peak_numbers:
        (times,
         widths,
         pmt_r ) = build_pmt_responses (s2df   [s2df   .peak == peak_number],
                                        s2pmtdf[s2pmtdf.peak == peak_number])
        sipm_r   = build_sipm_responses(sidf   [sidf   .peak == peak_number])
        s2s.append(S2(times, widths, pmt_r, sipm_r))

    return s2s
