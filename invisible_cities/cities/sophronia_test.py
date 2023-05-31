import os
import tables as tb

from pytest import mark

from .. core.testing_utils import assert_tables_equality
from .  sophronia          import sophronia


def test_sophronia_runs(sophronia_config, config_tmpdir):
    path_out = os.path.join(config_tmpdir, 'sophronia_runs.h5')
    nevt_req = 1
    config   = dict(**sophronia_config)
    config.update(dict(file_out    = path_out,
                       event_range = nevt_req))

    cnt = sophronia(**config)
    assert cnt.events_in   == nevt_req


def test_sophronia_contains_all_tables(sophronia_config, config_tmpdir):
    path_out = os.path.join(config_tmpdir, "test_sophronia_contains_all_tables.h5")
    nevt_req = 1
    config   = dict(**sophronia_config)
    config.update(dict(file_out    = path_out,
                       event_range = nevt_req))
    sophronia(**config)

    with tb.open_file(path_out) as h5out:
        assert "MC"                   in h5out.root
        assert "MC/hits"              in h5out.root
        assert "MC/configuration"     in h5out.root
        assert "MC/event_mapping"     in h5out.root
        assert "MC/hits"              in h5out.root
        assert "MC/particles"         in h5out.root
        assert "MC/sns_positions"     in h5out.root
        assert "MC/sns_response"      in h5out.root
        assert "DST/Events"           in h5out.root
        assert "RECO/Events"          in h5out.root
        assert "Run"                  in h5out.root
        assert "Run/eventMap"         in h5out.root
        assert "Run/events"           in h5out.root
        assert "Run/runInfo"          in h5out.root
        assert "Filters/s12_selector" in h5out.root
        assert "Filters/valid_hit"    in h5out.root


@mark.slow
def test_sophronia_exact_result(sophronia_config, Th228_hits, config_tmpdir):
    path_out = os.path.join(config_tmpdir, 'test_sophronia_exact_result.h5')
    config   = dict(**sophronia_config)
    config.update(dict(file_out = path_out))

    sophronia(**config)

    tables = ( "MC/hits", "MC/particles"
             , "DST/Events"
             , "RECO/Events"
             , "Run/events", "Run/runInfo"
             , "Filters/s12_selector", "Filters/valid_hit"
             )

    with tb.open_file(Th228_hits)   as true_output_file:
        with tb.open_file(path_out) as      output_file:
            for table in tables:
                assert hasattr(output_file.root, table), table
                got      = getattr(     output_file.root, table)
                expected = getattr(true_output_file.root, table)
                assert_tables_equality(got, expected)


def test_sophronia_filters_pmaps(sophronia_config, config_tmpdir):
    path_out = os.path.join(config_tmpdir, 'test_sophronia_filters_pmaps.h5')
    config   = dict(**sophronia_config)
    config.update(dict( file_out    = path_out
                      , event_range = (5, 8)))

    cnt = sophronia(**config)

    assert cnt.events_in          == 3
    assert cnt.events_out         == 1
    assert cnt.evtnum_list        == [400076]
    assert cnt.selection.n_passed == 1
    assert cnt.selection.n_failed == 2


@mark.skip(reason = "need a new test file for this")
def test_sophronia_filters_events_with_only_nn_hits():
    pass
