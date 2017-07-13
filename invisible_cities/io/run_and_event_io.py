from ..evm import nh5           as table_formats
from ..reco import tbl_functions as tbl

def _make_run_event_tables(hdf5_file, compression):

    c = tbl.filters(compression)
    rungroup = hdf5_file.create_group(hdf5_file.root, "Run")

    RunInfo, EventInfo = table_formats.RunInfo, table_formats.EventInfo
    MKT = hdf5_file.create_table

    run_info   = MKT(rungroup, "runInfo",   RunInfo,   "run info table", c)
    event_info = MKT(rungroup,  "events", EventInfo, "event info table", c)
    run_tables = (run_info, event_info)

    return run_tables


def run_and_event_writer(file, *, compression='ZLIB4'):
    run_tables = _make_run_event_tables(file, compression)
    def write_run_and_event(run_number, event_number, timestamp):
        run_table_dumper  (run_tables[0],   run_number)
        event_table_dumper(run_tables[1], event_number, timestamp)
    return write_run_and_event


def run_table_dumper(table, run_number):
    row = table.row
    row['run_number'] = run_number
    row.append()


def event_table_dumper(table, event_number, timestamp):
    row = table.row
    row["evt_number"] = event_number
    row["timestamp"] = timestamp
    row.append()
