import tables as tb

import invisible_cities.reco.nh5 as table_formats
import invisible_cities.reco.tbl_functions as tbl


class pmap_writer:

    def __init__(self, filename, compression = 'ZLIB4'):
        self._hdf5_file = tb.open_file(filename, 'w')
        self._tables = _make_tables(self._hdf5_file, compression)

    def __call__(self, event_number, s1, s2, s2si):
        s1  .store(self._tables[0], event_number)
        s2  .store(self._tables[1], event_number)
        s2si.store(self._tables[2], event_number)

    def close(self):
        self._hdf5_file.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


def _make_tables(hdf5_file, compression):

    c = tbl.filters(compression)

    pmaps_group = hdf5_file.create_group(hdf5_file.root, 'PMAPS')

    s1   = hdf5_file.create_table(pmaps_group, 'S1'  , S12 .table_format,   "S1 Table", c)
    s2   = hdf5_file.create_table(pmaps_group, 'S2'  , S12 .table_format,   "S2 Table", c)
    s2si = hdf5_file.create_table(pmaps_group, 'S2Si', S2Si.table_format, "S2Si Table", c)

    all_tables = (s1, s2, s2si)

    for table in all_tables:
        table.cols.event.create_index()

    return all_tables


# I think that an event is made up of a collection of S1s, S2s. The 

# I think that this is something like the (integrated) set of data
# recorded by the PMTs in single event ... with two instances per
# event: one for the S1s, one for the S2s
class S12: # Need a better name

    def __init__(self, data):
        # Temporary solution, can do better
        self._data = data

    # Other S12 utilities go here, for example statistical functions

    table_format = table_formats.S12

    def store(self, table, event_number):
        row = table.row
        for peak_number, (ts, Es) in self._data.items():
            assert len(ts) == len(Es)
            for t, E in zip(ts, Es):
                row["event"] = event_number
                row["peak"]  =  peak_number
                row["time"]  = t
                row["ene"]   = E
                row.append()

# I think that tis is something like the set of data recorded by the
# SiPMs during a single event ... only used for S2s
class S2Si: # Need a better name

    def __init__(self, data):
        # Temporary solution, can do better
        self._data = data

    # Other S2si utilities go here, for example statistical functions

    table_format = table_formats.S2Si

    def store(self, table, event_number):
        row = table.row
        for peak_number, sipms in self._data.items():
            for nsipm, Es in sipms:
                for nsample, E in enumerate(Es):
                    row["event"]   = event_number
                    row["peak"]    =  peak_number
                    row["nsipm"]   = nsipm
                    row["nsample"] = nsample
                    row["ene"]     = E
                    row.append()
