from functools import partial

import tables as tb

from .. evm                import nh5     as table_formats
from .. reco.tbl_functions import filters as tbl_filters
from .. io  .table_io      import make_table


def store_trigger(tables, trg_type, trg_channels):
    trg_type_table, trg_channels_array = tables

    if trg_type:
        trg_type_row   = trg_type_table.row
        trg_type_row['trigger_type'] = trg_type
        trg_type_row.append()

    if trg_channels is not None:
        new_shape = 1, trg_channels.shape[0] #add event dimension
        trg_channels_array.append(trg_channels.reshape(new_shape))


def trigger_writer(file, n_sensors, compression="ZLIB4"):
    tables = _make_tables(file, n_sensors, compression=compression)
    return partial(store_trigger, tables)


def _make_tables(hdf5_file, n_sensors, compression="ZLIB4"):
    compr         = tbl_filters(compression)
    trigger_group = hdf5_file.create_group(hdf5_file.root, 'Trigger')
    make_table    = partial(hdf5_file.create_table, trigger_group, filters=compr)

    trg_type    = make_table('trigger', table_formats.TriggerType, "Trigger Type")

    array_name = "events"
    trg_channels = hdf5_file.create_earray(trigger_group,
                                   array_name,
                                   atom    = tb.Int16Atom(),
                                   shape   = (0, n_sensors),
                                   filters = compr)

    trg_tables = trg_type, trg_channels

    return trg_tables


def trigger_dst_writer(hdf5_file, **kwargs):#{
    trigger_table = make_table(hdf5_file,
                               group       = "Trigger",
                               name        = "DST"    ,
                               fformat     = table_formats.TriggerTable,
                               description = "Simulated trigger data",
                               compression = 'ZLIB4')
    def write_trigger(trigger_info):#{
        row = trigger_table.row
        row["event"        ] = trigger_info.event
        row["pmt"          ] = trigger_info.pmt
        row["trigger_time" ] = trigger_info.trigger_time
        row["q"            ] = trigger_info.q
        row["width"        ] = trigger_info.width
        row["height"       ] = trigger_info.height
        row["valid_q"      ] = trigger_info.valid_q
        row["valid_w"      ] = trigger_info.valid_w
        row["valid_h"      ] = trigger_info.valid_h
        row["valid_peak"   ] = trigger_info.valid_peak
        row["valid_all"    ] = trigger_info.valid_all
        row["baseline"     ] = trigger_info.baseline
        row["max_height"   ] = trigger_info.max_height
        row["n_coinc"      ] = trigger_info.n_coinc
        row["closest_ttime"] = trigger_info.closest_ttime
        row["closest_pmt"  ] = trigger_info.closest_pmt
        row.append()

    def write_triggers(triggers):
        for t in triggers: write_trigger(t)

    return write_triggers
