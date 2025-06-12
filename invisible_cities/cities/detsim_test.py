import os
import shutil
import numpy  as np
import pandas as pd
import tables as tb

from .. cities.detsim  import detsim
from .. core           import system_of_units as units
from .. core.configure import configure
from .. io.dst_io      import load_dst
from .. types.symbols  import all_events
from .. core.testing_utils import assert_tables_equality
from .. core.testing_utils import ignore_warning

@ignore_warning.no_config_group
def test_detsim_contains_all_tables(ICDATADIR, output_tmpdir):

    PATH_IN  = os.path.join(ICDATADIR    , "nexus_new_kr83m_fast.newformat.sim.h5")
    PATH_OUT = os.path.join(output_tmpdir, "contain_all_tables.buffers.h5")
    conf = configure('detsim $ICTDIR/invisible_cities/config/detsim.conf'.split())
    conf.update(dict(files_in      = PATH_IN,
                     file_out      = PATH_OUT,
                     run_number    = 0,
                     event_range   = (0, 1)))
    result = detsim(**conf)
    buffer_params = conf["buffer_params"]

    with tb.open_file(PATH_OUT, mode="r") as h5out:
        assert hasattr(h5out.root,                  'MC')
        assert hasattr(h5out.root,                 'Run')
        assert hasattr(h5out.root,               'pmtrd')
        assert hasattr(h5out.root,              'sipmrd')
        assert hasattr(h5out.root, 'Filters/active_hits')

        pmtrd  = h5out.root.pmtrd
        sipmrd = h5out.root.sipmrd

        n_pmt  = int(buffer_params["length"] / buffer_params["pmt_width"] )
        n_sipm = int(buffer_params["length"] / buffer_params["sipm_width"])
        assert pmtrd .shape == (1,   12, n_pmt)
        assert sipmrd.shape == (1, 1792, n_sipm)


@ignore_warning.no_config_group
@ignore_warning.delayed_hits
def test_detsim_filter_active_hits(ICDATADIR, output_tmpdir):
    # the first event in test file labels are set to NO_ACTIVE
    PATH_IN  = os.path.join(ICDATADIR    , "nexus_new_kr83m_fast.newformat.sim.noactive.h5")
    PATH_OUT = os.path.join(output_tmpdir, "filtered_active_hits.buffers.h5")
    conf = configure('detsim $ICTDIR/invisible_cities/config/detsim.conf'.split())
    conf.update(dict(files_in    = PATH_IN,
                     file_out    = PATH_OUT,
                     run_number  = 0))
    result = detsim(**conf)

    assert result.events_in   == 2
    assert result.evtnum_list == [1]

    with tb.open_file(PATH_OUT, mode="r") as h5out:
        filters = h5out.root.Filters.active_hits.read()
        np.testing.assert_array_equal(filters["passed"], [False, True])


@ignore_warning.no_config_group
@ignore_warning.delayed_hits
def test_detsim_filter_dark_events(ICDATADIR, output_tmpdir):
    # this file contains delayed hits that are filtered out
    # leaving a very low energy hit that does not produce electrons
    # this test shows that without filtering out these events the city crashes
    PATH_IN  = os.path.join(ICDATADIR    , "nexus_not_enough_energy.h5")
    PATH_OUT = os.path.join(output_tmpdir, "filtered_dark_events.h5")
    conf = configure('detsim $ICTDIR/invisible_cities/config/detsim.conf'.split())
    conf.update(dict(files_in    = PATH_IN,
                     file_out    = PATH_OUT,
                     run_number  = 0))
    conf["buffer_params"]["max_time"] = 3 * units.ms

    result = detsim(**conf)

    assert result.events_in   == 1
    assert result.evtnum_list == []

    with tb.open_file(PATH_OUT, mode="r") as h5out:
        filters = h5out.root.Filters.dark_events.read()
        np.testing.assert_array_equal(filters["passed"], [False])


@ignore_warning.no_config_group
def test_detsim_filter_empty_waveforms(ICDATADIR, output_tmpdir):
    # the first event radius is slighty above NEW active radius of 208.0 mm
    PATH_IN  = os.path.join(ICDATADIR, "nexus_new_kr83m_fast.newformat.sim.emptywfs.h5")
    PATH_OUT = os.path.join(output_tmpdir, "filtered_empty_waveforms.buffers.h5")
    conf = configure('detsim $ICTDIR/invisible_cities/config/detsim.conf'.split())
    physics_params = conf["physics_params"]
    physics_params["transverse_diffusion"] = 0 * units.mm / units.cm**0.5
    conf.update(dict(files_in    = PATH_IN,
                     file_out    = PATH_OUT,
                     run_number  = 0,
                     physics_params = physics_params))
    result = detsim(**conf)

    assert result.events_in   == 2
    assert result.evtnum_list == [1]

    with tb.open_file(PATH_OUT, mode="r") as h5out:
        filters = h5out.root.Filters.signal.read()
        np.testing.assert_array_equal(filters["passed"], [False, True])


@ignore_warning.no_config_group
def test_detsim_empty_input_file(ICDATADIR, output_tmpdir):

    PATH_IN  = os.path.join(ICDATADIR    , "empty_mcfile.h5")
    PATH_OUT = os.path.join(output_tmpdir, "empty_imput_file.buffers.h5")
    conf = configure('detsim $ICTDIR/invisible_cities/config/detsim.conf'.split())
    conf.update(dict(files_in    = PATH_IN,
                     file_out    = PATH_OUT,
                     run_number  = 0))
    result = detsim(**conf)

    assert result.events_in   == 0
    assert result.evtnum_list == []


@ignore_warning.no_config_group
def test_detsim_exact(ICDATADIR, output_tmpdir):

    PATH_IN   = os.path.join(ICDATADIR    , "nexus_new_kr83m_fast.newformat.sim.h5")
    PATH_OUT  = os.path.join(output_tmpdir, "exact_tables.buffers.h5")
    PATH_TRUE = os.path.join(ICDATADIR    , "detsim_new_kr83m_fast.newformat.sim.h5")
    conf = configure('detsim $ICTDIR/invisible_cities/config/detsim.conf'.split())
    conf.update(dict(files_in      = PATH_IN,
                     file_out      = PATH_OUT,
                     run_number    = 0,
                     event_range   = all_events,
                     buffer_params = dict(length      = 800 * units.mus,
                                          pmt_width   = 100 * units.ns ,
                                          sipm_width  =   1 * units.mus,
                                          max_time    =  10 * units.ms ,
                                          pre_trigger = 100 * units.mus,
                                          trigger_thr = 0)))
    np.random.seed(1234)
    result = detsim(**conf)

    tables = ["pmtrd", "sipmrd",
              "Run/eventMap", "Run/events", "Run/runInfo",
              "MC/configuration", "MC/event_mapping", "MC/hits",
              "MC/particles", "MC/sns_positions", "MC/sns_response",
              "Filters/active_hits"]

    with tb.open_file(PATH_TRUE) as true_output_file:
        with tb.open_file(PATH_OUT) as output_file:
            for table in tables:
                assert hasattr(output_file.root, table)
                got      = getattr(     output_file.root, table)
                expected = getattr(true_output_file.root, table)
                assert_tables_equality(got, expected)


@ignore_warning.no_config_group
def test_detsim_exact_time_translation(ICDATADIR, output_tmpdir):

    PATH_IN   = os.path.join(ICDATADIR    , "nexus_new_kr83m_fast.newformat.sim.h5")
    PATH_OUT  = os.path.join(output_tmpdir, "exact_time_translation.buffers.h5")
    # modified input file: time --> time + 10 micro-seconds
    modified_testfile = os.path.join(ICDATADIR, "nexus_new_kr83m_fast.newformat.sim.time10mus.h5")

    # run over test file
    conf = configure('detsim $ICTDIR/invisible_cities/config/detsim.conf'.split())
    conf.update(dict(files_in      = PATH_IN,
                     file_out      = PATH_OUT,
                     run_number    = 0,
                     event_range   = all_events))
    np.random.seed(1234)
    result = detsim(**conf)

    # run over modified file
    modified_file_out = os.path.join(output_tmpdir, "detsim_test_modified.h5")
    conf = configure('detsim $ICTDIR/invisible_cities/config/detsim.conf'.split())
    conf.update(dict(files_in      = modified_testfile,
                     file_out      = modified_file_out,
                     run_number    = 0,
                     event_range   = all_events))
    np.random.seed(1234)
    result = detsim(**conf)

    tables = ["pmtrd", "sipmrd",
              "Run/eventMap", "Run/events", "Run/runInfo",
              "MC/configuration", "MC/event_mapping", #"MC/hits", note that MC/hits are different
              "MC/particles", "MC/sns_positions", "MC/sns_response",
              "Filters/active_hits"]

    with tb.open_file(PATH_OUT) as output_file:
        with tb.open_file(modified_file_out) as modified_output_file:
            for table in tables:
                assert hasattr(output_file.root, table)
                got      = getattr( modified_output_file.root, table)
                expected = getattr(          output_file.root, table)
                assert_tables_equality(got, expected)


@ignore_warning.no_config_group
def test_detsim_buffer_times(ICDATADIR, output_tmpdir):
    """This test checks that the signal is properly placed in the buffer.
    In particular:
      - the S1 start time is positioned at pretrigger
      - the S2 start time is positioned at pretrigger + min(time + Z/dv)
      - the S2   end time is positioned at pretrigger + max(time + Z/dv)
    This can only be tested with nulls longinutinal diffusion and el_drift_velocity.
    To simplify the S2 search, we assume that it is beyond 2 micro-seconds from S1,
    then we must use an event at Z drift larger than that.
    """

    PATH_IN   = os.path.join(ICDATADIR    , "nexus_new_kr83m_fast.newformat.sim.h5")
    PATH_OUT  = os.path.join(output_tmpdir, "buffer_times.buffers.h5")

    # run over test file
    conf = configure('detsim $ICTDIR/invisible_cities/config/detsim.conf'.split())
    physics_params = conf["physics_params"]
    physics_params["longitudinal_diffusion"] = 0       * units.mm / units.cm**0.5
    physics_params["el_drift_velocity"]      = 1000000 * units.mm / units.mus
    # take second event because it has a relatively large Z
    conf.update(dict(files_in      = PATH_IN,
                     file_out      = PATH_OUT,
                     run_number    = 0,
                     event_range   = (1, 2),
                     physics_params= physics_params))
    result = detsim(**conf)

    buffer_params = conf["buffer_params"]
    # compute max signal time
    hits = load_dst(PATH_IN, "MC", "hits")
    hits = hits[hits["event_id"]==1]
    z, time  = hits.z.values, hits.time.values
    s2_times = z/physics_params["drift_velocity"] + time

    s2_tmin = min(s2_times) + buffer_params["pre_trigger"]
    s2_tmax = max(s2_times) + buffer_params["pre_trigger"]

    with tb.open_file(PATH_OUT) as h5out:
        pmtwf  = h5out.root.pmtrd .read()[0].sum(axis=0)
        sipmwf = h5out.root.sipmrd.read()[0].sum(axis=0)

    pmt_signal_bins  = np.argwhere(pmtwf >0).flatten()
    sipm_signal_bins = np.argwhere(sipmwf>0).flatten()

    # PMTs
    # start s1 time
    tstart = pmt_signal_bins[0]*buffer_params["pmt_width"]
    assert tstart == buffer_params["pre_trigger"]

    # search S2, assume that it is above pretrigger + 2 micro-s
    pre_idx = np.floor(buffer_params["pre_trigger"]/buffer_params["pmt_width"])
    s2_init = pre_idx + np.floor(2*units.mus/buffer_params["pmt_width"])
    s2_bins = pmt_signal_bins[pmt_signal_bins>s2_init]

    tstart = s2_bins[0]  * buffer_params["pmt_width"]
    tend   = s2_bins[-1] * buffer_params["pmt_width"]

    assert tstart == np.floor(s2_tmin / buffer_params["pmt_width"]) * buffer_params["pmt_width"]
    assert tend   == np.floor(s2_tmax / buffer_params["pmt_width"]) * buffer_params["pmt_width"]

    # SIPMs
    # search S2, assume that it is above pretrigger + 2 micro-s
    pre_idx = np.floor(buffer_params["pre_trigger"]/buffer_params["sipm_width"])
    s2_init = pre_idx + np.floor(2*units.mus/buffer_params["sipm_width"])
    s2_bins = sipm_signal_bins[sipm_signal_bins>s2_init]

    tstart = s2_bins[0]  * buffer_params["sipm_width"]
    tend   = s2_bins[-1] * buffer_params["sipm_width"]

    assert tstart == np.floor(s2_tmin / buffer_params["sipm_width"]) * buffer_params["sipm_width"]
    assert tend   == np.floor(s2_tmax / buffer_params["sipm_width"]) * buffer_params["sipm_width"]


@ignore_warning.no_config_group
def test_detsim_hits_without_strings(ICDATADIR, output_tmpdir):
    PATH_IN  = os.path.join(ICDATADIR    , "nexus_next100_nostrings.h5")
    PATH_OUT = os.path.join(output_tmpdir, "detsim_nostrings.h5")
    conf = configure('detsim $ICTDIR/invisible_cities/config/detsim_next100.conf'.split())
    conf.update(dict(files_in    = PATH_IN,
                     file_out    = PATH_OUT,
                     run_number  = 0))
    result = detsim(**conf)
    assert result.evtnum_list == [0, 1]

    df = pd.read_hdf(PATH_OUT, 'MC/string_map')
    assert not df.empty
