import tables as tb

from   invisible_cities.reco.nh5 import S12, S2Si
import invisible_cities.reco.tbl_functions as tbl


class pmap_writer:

    def __init__(self, filename, compression = 'ZLIB4'):
        self._hdf5_file = tb.open_file(filename, 'w')
        self._tables = _make_tables(self._hdf5_file, compression)

    def __call__(self, event_number, s1, s2, s2si):
        _store_s12 (s1,   self._tables[0], event_number)
        _store_s12 (s2,   self._tables[1], event_number)
        _store_s2si(s2si, self._tables[2], event_number)

    def close(self):
        self._hdf5_file.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


def _make_tables(hdf5_file, compression):

    c = tbl.filters(compression)

    pmaps_group = hdf5_file.create_group(hdf5_file.root, 'PMAPS')

    s1   = hdf5_file.create_table(pmaps_group, 'S1'  , S12 ,   "S1 Table", c)
    s2   = hdf5_file.create_table(pmaps_group, 'S2'  , S12 ,   "S2 Table", c)
    s2si = hdf5_file.create_table(pmaps_group, 'S2Si', S2Si, "S2Si Table", c)

    all_tables = (s1, s2, s2si)

    for table in all_tables:
        table.cols.event.create_index()

    return all_tables


def _store_s12(event, table, event_number):
    row = table.row
    for peak_number, (ts, Es) in event.items():
        assert len(ts) == len(Es)
        for t, E in zip(ts, Es):
            row["event"] = event_number
            row["peak"]  =  peak_number
            row["time"]  = t
            row["ene"]   = E
            row.append()


def _store_s2si(event, table, event_number):
    row = table.row
    for peak_number, sipms in event.items():
        for nsipm, Es in sipms:
            for nsample, E in enumerate(Es):
                row["event"]   = event_number
                row["peak"]    =  peak_number
                row["nsipm"]   = nsipm
                row["nsample"] = nsample
                row["ene"]     = E
                row.append()
