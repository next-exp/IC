import tables as tb
import pandas as pd

from .. evm      import nh5        as table_formats
from .  table_io import make_table


def event_filter_writer(file, filter_name, *, compression='ZLIB4'):
    table = make_table(file,
                       group       = "Filters",
                        name        = filter_name,
                       fformat     = table_formats.EventPassedFilter,
                       description = "Event has passed filter flag",
                       compression = compression)

    def write_event_passed(event_number, passed_filter):
        table.row["event" ] = event_number
        table.row["passed"] = passed_filter
        table.row.append()

    return write_event_passed


def event_filter_reader(filename):
    def read_as_df(table):
        df = pd.DataFrame.from_records(table.read(), index="event")
        df.rename(columns=dict(passed = table.name), inplace=True)
        return df

    with tb.open_file(filename) as file:
        tables  = map(read_as_df, file.root.Filters)
        full_df = pd.concat(list(tables), axis=1)
        return full_df.fillna(False)
