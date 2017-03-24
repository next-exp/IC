import tables as tb

import invisible_cities.reco.nh5 as table_formats
import invisible_cities.reco.tbl_functions as tbl


class pmap_writer:

    def __init__(self, filename, compression = 'ZLIB4'):
        self._hdf5_file = tb.open_file(filename, 'w')
        self._run_tables = _make_run_event_tables(self._hdf5_file,
                                                          compression)
        self._pmp_tables = _make_pmp_tables(      self._hdf5_file,
                                                          compression)

    def __call__(self, run_number, event_number, timestamp, s1, s2, s2si):
        s1  .store(self._pmp_tables[0], event_number)
        s2  .store(self._pmp_tables[1], event_number)
        s2si.store(self._pmp_tables[2], event_number)
        run_writer(self._run_tables[0], run_number)
        event_writer(self._run_tables[1], event_number, timestamp)

    def close(self):
        self._hdf5_file.close()

    @property
    def file(self):
        return self._hdf5_file

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


def _make_run_event_tables(hdf5_file, compression):

    c = tbl.filters(compression)
    rungroup     = hdf5_file.create_group(hdf5_file.root, "Run")

    RunInfo, EventInfo = table_formats.RunInfo, table_formats.EventInfo
    MKT = hdf5_file.create_table

    run_info   = MKT(rungroup, "runInfo",   RunInfo,   "run info table", c)
    event_info = MKT(rungroup,  "events", EventInfo, "event info table", c)
    run_tables = (run_info, event_info)

    return run_tables

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


def run_writer(table, run_number):
    row = table.row
    row['run_number'] = run_number
    row.append()

def event_writer(table, event_number, timestamp):
    row = table.row
    row["evt_number"] = event_number
    row["timestamp"] = timestamp
    row.append()

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
