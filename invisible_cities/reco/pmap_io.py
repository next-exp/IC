import tables as tb

from . import nh5           as table_formats
from . import tbl_functions as tbl


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
    s1         = MKT(pmaps_group, 'S1'  , S12 .table_format,   "S1 Table", c)
    s2         = MKT(pmaps_group, 'S2'  , S12 .table_format,   "S2 Table", c)
    s2si       = MKT(pmaps_group, 'S2Si', S2Si.table_format, "S2Si Table", c)

    pmp_tables = (s1, s2, s2si)

    for table in pmp_tables:
        table.cols.event.create_index()

    return pmp_tables


class S12(dict):
    """Defines an S12 type."""

    table_format = table_formats.S12

    def store(self, table, event_number):
        row = table.row
        for peak_number, (ts, Es) in self.items():
            assert len(ts) == len(Es)
            for t, E in zip(ts, Es):
                row["event"] = event_number
                row["peak"]  =  peak_number
                row["time"]  = t
                row["ene"]   = E
                row.append()

class S2Si(dict):
    """Defines an S2Si type."""

    table_format = table_formats.S2Si

    def store(self, table, event_number):
        row = table.row
        for peak, sipm in self.items():
            for nsipm, ene in sipm.items():
                for E in ene:
                    row["event"]   = event_number
                    row["peak"]    = peak
                    row["nsipm"]   = nsipm
                    #row["nsample"] = nsample
                    row["ene"]     = E
                    row.append()
