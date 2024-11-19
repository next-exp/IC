import os
import tables as tb

from pytest import mark

from .. core.testing_utils   import assert_tables_equality
from .. core.testing_utils   import ignore_warning
from .. core.system_of_units import pes
from .  sophronia            import sophronia


@ignore_warning.no_config_group
def test_sophronia_runs(sophronia_config, config_tmpdir):
    path_out = os.path.join(config_tmpdir, 'sophronia_runs.h5')
    nevt_req = 1
    config   = dict(**sophronia_config)
    config.update(dict(file_out    = path_out,
                       event_range = nevt_req))

    cnt = sophronia(**config)
    assert cnt.events_in   == nevt_req


@ignore_warning.no_config_group
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


@ignore_warning.no_config_group
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


@ignore_warning.no_config_group
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


@ignore_warning.no_config_group
def test_sophronia_filters_events_with_only_nn_hits(config_tmpdir, sophronia_config):
    """
    Run with a high q threshold so all hits are NN-hits.
    Check that these events are filtered out.
    """
    path_out = os.path.join(config_tmpdir, 'test_sophronia_filters_events_with_only_nn_hits.h5')
    config   = dict(**sophronia_config)
    config.update(dict( file_out    = path_out
                      , q_thr       = 1e4 * pes
                      , event_range = 1 ))

    sophronia(**config)

    with tb.open_file(config["files_in"]) as input_file:
        event_number = input_file.root.Run.events[0][0]

    with tb.open_file(path_out) as output_file:
        # Check that the event passes the s12_selector, which is
        # applied earlier. Then check it doesn't pass the valid_hit
        # filter, which checks that there are at least 1 non-NN hit.
        # Each entry in these tables has the form
        # (event_number, passed_flag)
        assert     output_file.root.Filters.s12_selector[0][1]
        assert not output_file.root.Filters.valid_hit   [0][1]


@ignore_warning.no_config_group
def test_sophronia_keeps_hitless_events(config_tmpdir, sophronia_config):
    """
    Run with a high q threshold so all hits are discarded (turned into NN).
    Check that these events are still in the /Run/events output, but not in
    the /RECO/events output.
    """
    path_out = os.path.join(config_tmpdir, 'test_sophronia_keeps_hitless_events.h5')
    config   = dict(**sophronia_config)
    config.update(dict( file_out    = path_out
                      , q_thr       = 1e4 * pes
                      , event_range = 1 ))

    sophronia(**config)

    with tb.open_file(config["files_in"]) as input_file:
        event_number = input_file.root.Run.events[0][0]

    with tb.open_file(path_out) as output_file:
        assert len(output_file.root.Run.events) == 1
        assert event_number == output_file.root.Run.events[0][0]
        assert event_number not in output_file.root.RECO.Events.col("event")
