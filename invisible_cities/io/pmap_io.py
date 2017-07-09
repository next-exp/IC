import tables as tb
import pandas as pd

from .. reco import nh5           as table_formats
from .. reco import tbl_functions as tbl
#from .. reco.pmaps_functions_c      import df_to_pmaps_dict
from .. reco.pmaps_functions_c      import df_to_s1_dict
from .. reco.pmaps_functions_c      import df_to_s2_dict
from .. reco.pmaps_functions_c      import df_to_s2si_dict

def s1_s2_si_from_pmaps(s1_dict, s2_dict, s2si_dict, evt_number):
    s1 = s1_dict  .get(evt_number, None)
    s2 = s2_dict  .get(evt_number, None)
    s2si = s2si_dict.get(evt_number, None)
    return s1, s2, s2si

def load_pmaps(PMP_file_name):
    """Read the PMAP file and return transient PMAP rep."""

    s1t, s2t, s2sit = read_pmaps(PMP_file_name)
    S1              = df_to_s1_dict(s1t)
    S2              = df_to_s2_dict(s2t)
    S2Si            = df_to_s2si_dict(s2t, s2sit)
    return S1, S2, S2Si


def read_pmaps(PMP_file_name):
    """Return the PMAPS as PD DataFrames."""
    with tb.open_file(PMP_file_name, 'r') as h5f:
        s1t   = h5f.root.PMAPS.S1
        s2t   = h5f.root.PMAPS.S2
        s2sit = h5f.root.PMAPS.S2Si

        return (pd.DataFrame.from_records(s1t  .read()),
                pd.DataFrame.from_records(s2t  .read()),
                pd.DataFrame.from_records(s2sit.read()))


def read_run_and_event_from_pmaps_file(PMP_file_name):
    """Return the PMAPS as PD DataFrames."""
    with tb.open_file(PMP_file_name, 'r') as h5f:
        event_t = h5f.root.Run.events
        run_t   = h5f.root.Run.runInfo

        return (pd.DataFrame.from_records(run_t  .read()),
                pd.DataFrame.from_records(event_t.read()))



def pmap_writer(file, *, compression='ZLIB4'):
    pmp_tables = _make_pmp_tables(file, compression)
    def write_pmap(event_number, s1, s2, s2si):
        s1  .store(pmp_tables[0], event_number)
        s2  .store(pmp_tables[1], event_number)
        s2si.store(pmp_tables[2], event_number)
    return write_pmap


def _make_pmp_tables(hdf5_file, compression):

    c = tbl.filters(compression)
    pmaps_group  = hdf5_file.create_group(hdf5_file.root, 'PMAPS')
    MKT = hdf5_file.create_table
    s1         = MKT(pmaps_group, 'S1'  , table_formats.S12,   "S1 Table", c)
    s2         = MKT(pmaps_group, 'S2'  , table_formats.S12,   "S2 Table", c)
    s2si       = MKT(pmaps_group, 'S2Si', table_formats.S2Si, "S2Si Table", c)

    pmp_tables = (s1, s2, s2si)

    for table in pmp_tables:
        table.cols.event.create_index()

    return pmp_tables
