import os
import shutil
import numpy  as np
import pandas as pd
import tables as tb

from .. cities.paolina import paolina
from .. core           import system_of_units as units
from .. core.configure import configure
from .. core.configure import all             as all_events
from .. io.dst_io      import load_dst

from .. core.testing_utils   import assert_tables_equality


def test_paolina_contains_all_tables(ICDATADIR, output_tmpdir):

    PATH_IN  = os.path.join(ICDATADIR    , "deconvoluted_0nubb_next100.h5")
    PATH_OUT = os.path.join(output_tmpdir, "contain_all_tables.paolina.h5")
    conf = configure('paolina $ICTDIR/invisible_cities/config/paolina.conf'.split())
    conf.update(dict(files_in      = PATH_IN,
                     file_out      = PATH_OUT,
                     run_number    = 0,
                     event_range   = (0, 2)))
    result = paolina(**conf)

    with tb.open_file(PATH_OUT, mode="r") as h5out:
        assert hasattr(h5out.root,               'MC')
        assert hasattr(h5out.root,  'Tracking/Tracks')
        assert hasattr(h5out.root,              'Run')
        assert hasattr(h5out.root, 'Filters/one_peak')


def test_paolina_filter_multipeak(ICDATADIR, output_tmpdir):

    PATH_IN  = os.path.join(ICDATADIR    , "deconvoluted_0nubb_next100.h5")
    PATH_OUT = os.path.join(output_tmpdir, "filtered_multipeak_events.h5")
    conf = configure('paolina $ICTDIR/invisible_cities/config/paolina.conf'.split())
    conf.update(dict(files_in      = PATH_IN,
                     file_out      = PATH_OUT,
                     run_number    = 0,
                     event_range   = all_events))
    result = paolina(**conf)

    assert result.events_in   == 10
    assert result.evtnum_list == [2, 4, 6, 10, 12, 14, 16, 18]
    passed = [False,  True,  True,  True, False,  True,  True,  True,  True, True]

    with tb.open_file(PATH_OUT, mode="r") as h5out:
        filters = h5out.root.Filters.one_peak.read()
        np.testing.assert_array_equal(filters["passed"], passed)


def test_paolina_empty_input_file(ICDATADIR, output_tmpdir):

    PATH_IN  = os.path.join(ICDATADIR    , "empty_file.h5")
    PATH_OUT = os.path.join(output_tmpdir, "empty_paolina.h5")
    conf = configure('paolina $ICTDIR/invisible_cities/config/paolina.conf'.split())
    conf.update(dict(files_in      = PATH_IN,
                     file_out      = PATH_OUT,
                     run_number    = 0,
                     event_range   = all_events))
    result = paolina(**conf)

    assert result.events_in   == 0
    assert result.evtnum_list == []


def test_paolina_exact(ICDATADIR, output_tmpdir):

    PATH_IN   = os.path.join(ICDATADIR    , "deconvoluted_0nubb_next100.h5")
    PATH_OUT  = os.path.join(output_tmpdir, "exact_tables.paolina.h5")
    PATH_TRUE = os.path.join(ICDATADIR    , "paolina_0nubb_next100_10evts.h5")
    conf = configure('paolina $ICTDIR/invisible_cities/config/paolina.conf'.split())
    conf.update(dict(files_in      = PATH_IN,
                     file_out      = PATH_OUT,
                     run_number    = 0,
                     event_range   = all_events))
    np.random.seed(1234)
    result = paolina(**conf)

    tables = ["Tracking/Tracks",
              "Run/eventMap", "Run/events", "Run/runInfo",
              "MC/configuration", "MC/event_mapping", "MC/hits",
              "MC/particles", "MC/sns_positions", "MC/sns_response",
              "Filters/one_peak"]

    with tb.open_file(PATH_TRUE) as true_output_file:
        with tb.open_file(PATH_OUT) as output_file:
            for table in tables:
                assert hasattr(output_file.root, table)
                got      = getattr(     output_file.root, table)
                expected = getattr(true_output_file.root, table)
                assert_tables_equality(got, expected)
