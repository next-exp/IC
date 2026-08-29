# io/amplification_io.py

import tables as tb


class AmplificationRow(tb.IsDescription):
    event   = tb.Int32Col(pos=0)
    channel = tb.Int16Col(pos=1)
    area_hg = tb.Float32Col(pos=2)
    area_lg = tb.Float32Col(pos=3)


def amplification_writer(h5out, group_name="AMPLIFICATION", table_name="fiber_amplification"):
    """
    Returns a writer with signature
        write(event_number, channels, areas_hg, areas_lg)
    appending one row per isolated peak. channels/areas_hg/areas_lg are
    equal-length 1D arrays -- the output of integrate_hg_lg_pairs_ercilia
    for a single event.
    """
    if group_name not in h5out.root:
        h5out.create_group(h5out.root, group_name)
    group = getattr(h5out.root, group_name)

    table = h5out.create_table(group, table_name, AmplificationRow,
                                "HG/LG paired peak areas")
    row = table.row

    def write_amplification(event_number, channels, areas_hg, areas_lg):
        for ch, a_hg, a_lg in zip(channels, areas_hg, areas_lg):
            row["event"]   = event_number
            row["channel"] = ch
            row["area_hg"] = a_hg
            row["area_lg"] = a_lg
            row.append()
        table.flush()

    return write_amplification