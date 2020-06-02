import os

from collections import namedtuple

from pytest import fixture
from pytest import mark

import numpy  as np
import tables as tb

from . event_filter_io import event_filter_writer
from . event_filter_io import event_filter_reader


filter_data = namedtuple("filter_data", "filename nevt tables evt passed")


@fixture(scope="session")
def one_filter_table_file(ICDATADIR):
    filename = os.path.join(ICDATADIR, "one_filter_table_file.h5")
    nevt     = 20
    tables   = "a_filter",
    evts     = np.arange(0, 40, 2)
    passed   = [True] * 3 + [False] * 4 + [True] * 3 + [False] * 5 + [True] * 5

    return filter_data(filename, nevt, tables, evts, np.array(passed))


@fixture(scope="session")
def multiple_filter_tables_file_same_length(ICDATADIR):
    filename = os.path.join(ICDATADIR, "multiple_filter_tables_file_same_length.h5")
    nevt     = 5
    tables   = "a_filter", "b_filter", "c_filter", "d_filter"
    evts     = np.arange(0, 15, 3)
    passed   = np.array([[ True, True , True , True ],
                         [ True, True , True , False],
                         [ True, True , False, False],
                         [ True, False, False, False],
                         [False, False, False, False]])

    return filter_data(filename, nevt, tables, evts, np.array(passed))


@fixture(scope="session")
def multiple_filter_tables_file_different_lengths(ICDATADIR):
    filename = os.path.join(ICDATADIR, "multiple_filter_tables_file_different_lengths.h5")
    nevt     = 5
    tables   = "a_filter", "b_filter", "c_filter", "d_filter"
    evts     = np.arange(0, 20, 4)
    passed   = np.array([[ True, True , True , True ],
                         [ True, True , True , False],
                         [ True, True , False, False],
                         [ True, False, False, False],
                         [False, False, False, False]])

    return filter_data(filename, nevt, tables, evts, np.array(passed))


@fixture(scope="session")
def multiple_filter_tables(request):
    return request.getfixturevalue(request.param)


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


@mark.parametrize("multiple_filter_tables",
                  ("multiple_filter_tables_file_same_length"      ,
                   "multiple_filter_tables_file_different_lengths"),
                  indirect=True)
def test_event_filter_reader_column_names(multiple_filter_tables):
    data = multiple_filter_tables
    df   = event_filter_reader(data.filename)
    assert sorted(df.columns.values) == sorted(data.tables)


def test_event_filter_reader_one_filter(one_filter_table_file):
    data = one_filter_table_file
    df   = event_filter_reader(data.filename)
    assert np.all(df.index   .values == data.evt   )
    assert np.all(df.a_filter.values == data.passed)


def test_event_filter_reader_multiple_filters_same_length(multiple_filter_tables_file_same_length):
    data = multiple_filter_tables_file_same_length
    df   = event_filter_reader(data.filename)

    assert np.all(df.index.values == data.evt)

    for column, expected in zip(data.tables, data.passed.T):
        assert np.all(getattr(df, column).values == expected)


def test_event_filter_reader_multiple_filters_different_lengths(multiple_filter_tables_file_different_lengths):
    data = multiple_filter_tables_file_different_lengths
    df   = event_filter_reader(data.filename)

    assert np.all(df.index.values == data.evt)

    for column, expected in zip(data.tables, data.passed.T):
        assert np.all(getattr(df, column).values == expected)
