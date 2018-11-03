import os

from pytest import mark

import numpy  as np
import tables as tb

from . event_filter_io import event_filter_writer


@mark.parametrize("table_names",
                  (("a_filter",),
                   ("a_filter", "b_filter"),
                   ("a_filter", "b_filter", "c_filter")))
def test_event_filter_writer_creates_group_and_tables(config_tmpdir, table_names):
    filename = os.path.join(config_tmpdir, f"test_event_filter_writer_creates_group_and_table.h5")

    with tb.open_file(filename, "w") as file:
        # do nothing but create the tables
        for table_name in table_names:
            event_filter_writer(file, table_name)

    with tb.open_file(filename, "r") as file:
        assert "Filters"  in file.root
        for table_name in table_names:
            assert table_name in file.root.Filters


def test_event_filter_writer_writes_data_correctly(config_tmpdir):
    filename = os.path.join(config_tmpdir, f"test_event_filter_writer_writes_data_correctly.h5")
    n_evt    = 10
    evt_nos  = np.random.randint(0, 10000, size=n_evt)
    passed   = np.random.randint(0,     2, size=n_evt).astype(bool)

    with tb.open_file(filename, "w") as file:
        write = event_filter_writer(file, "a_filter")

        for evt, value in zip(evt_nos, passed):
            write(evt, value)

    with tb.open_file(filename, "r") as file:
        data = file.root.Filters.a_filter
        assert data.nrows == n_evt
        for i, (evt_no, pss) in enumerate(data.read()):
            assert evt_no == evt_nos[i]
            assert pss    == passed [i]
