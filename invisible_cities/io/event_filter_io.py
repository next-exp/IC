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
