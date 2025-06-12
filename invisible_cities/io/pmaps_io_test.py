import os
import shutil
from collections import namedtuple
from collections import defaultdict

from pytest import mark
from pytest import approx
from pytest import fixture
from pytest import raises

import tables as tb
import numpy  as np
import pandas as pd

from ..core.exceptions    import InvalidInputFileStructure
from ..core.testing_utils import assert_PMap_equality
from ..core.testing_utils import assert_dataframes_equal
from ..core.testing_utils import exactly
from ..evm .pmaps         import S1
from ..evm .pmaps         import S2
from ..evm .pmaps         import PMTResponses
from ..evm .pmaps         import SiPMResponses
from ..evm .pmaps         import PMap
from .                    import pmaps_io as pmpio
from .                    import run_and_event_io as reio

from typing import Generator
from typing import Mapping


pmaps_data = namedtuple("pmaps_data", """evt_numbers      peak_numbers
                                         evt_numbers_pmt  peak_numbers_pmt
                                         evt_numbers_sipm peak_numbers_sipm
                                         times bwidths npmts nsipms
                                         enes enes_pmt enes_sipm""")

@fixture(scope="session")
def two_pmaps_evm():
    """
    Generate two pmaps (two events) with random data. We fix the RNG
    state for reproducibility. We make minimal changes to the sensor
    ID and wfs between pmts and sipms to keep this fixture to a
    minimum since the data values are irrelevant. We use different
    number of S1s and S2s to cover more cases.
    """
    state = np.random.get_state()
    np.random.seed(123456789)

    pmaps = {}
    for i in range(2):
        n_sensors = 5
        n_samples = 50
        n_s1      = 1 + i
        n_s2      = 2 - i

        peaks = []
        for with_sipms in [False]*n_s1 + [True]*n_s2:
            times   = np.arange     (n_samples, dtype=np.float32)
            bwidths = np.ones       (n_samples, dtype=np.float32)
            ids     = np.arange     (n_sensors) * 3
            wfs     = np.random.rand(n_sensors, n_samples).astype(np.float32)
            pmts    =  PMTResponses(ids     , wfs  )
            if with_sipms:
                sipms = SiPMResponses(ids+1000, wfs*2)
                peak  = S2
            else:
                sipms = SiPMResponses.build_empty_instance()
                peak  = S1
            peaks.append(peak(times, bwidths, pmts, sipms))

        s1s, s2s = peaks[:n_s1], peaks[n_s1:]
        pmaps[i*5] = PMap(s1s, s2s)

    np.random.set_state(state)
    return pmaps


@fixture(scope="session")
def two_pmaps_dfs(two_pmaps_evm):
    """Same as `two_pmaps` but with DataFrames"""
    s1_data = pmaps_to_arrays(two_pmaps_evm, list(two_pmaps_evm), "s1s")
    s2_data = pmaps_to_arrays(two_pmaps_evm, list(two_pmaps_evm), "s2s")

    s1 = pd.DataFrame(dict( event  = s1_data.evt_numbers
                          , time   = s1_data.times
                          , peak   = s1_data.peak_numbers.astype(np.uint8)
                          , bwidth = s1_data.bwidths
                          , ene    = s1_data.enes
                          ))
    s2 = pd.DataFrame(dict( event  = s2_data.evt_numbers
                          , time   = s2_data.times
                          , peak   = s2_data.peak_numbers.astype(np.uint8)
                          , bwidth = s2_data.bwidths
                          , ene    = s2_data.enes
                          ))
    si = pd.DataFrame(dict( event  = s2_data.evt_numbers_sipm
                          , peak   = s2_data.peak_numbers_sipm.astype(np.uint8)
                          , nsipm  = s2_data.nsipms.astype(np.int16)
                          , ene    = s2_data.enes_sipm
                          ))
    s2pmt = pd.DataFrame(dict( event = s2_data.evt_numbers_pmt
                             , peak  = s2_data.peak_numbers_pmt.astype(np.uint8)
                             , npmt  = s2_data.npmts.astype(np.uint8)
                             , ene   = s2_data.enes_pmt
                             ))
    s1pmt = pd.DataFrame(dict( event = s1_data.evt_numbers_pmt
                             , peak  = s1_data.peak_numbers_pmt.astype(np.uint8)
                             , npmt  = s1_data.npmts.astype(np.uint8)
                             , ene   = s1_data.enes_pmt
                             ))
    return s1, s2, si, s1pmt, s2pmt


@fixture(scope="session")
def two_pmaps(two_pmaps_evm, two_pmaps_dfs, output_tmpdir):
    pmap_filename = os.path.join(output_tmpdir, "two_pmaps.h5")
    run_number    = 0 # irrelevant
    timestamp     = 0 # irrelevant

    with tb.open_file(pmap_filename, "w") as output_file:
        write_pmap    = pmpio.pmap_writer(output_file)
        write_evtinfo = reio.run_and_event_writer(output_file)
        for event_number, pmap in two_pmaps_evm.items():
            write_pmap   (pmap, event_number)
            write_evtinfo(run_number, event_number, timestamp)

    return pmap_filename, two_pmaps_evm, two_pmaps_dfs


def pmaps_to_arrays(pmaps, evt_numbers, attr):
    data = defaultdict(list)
    for evt_number, pmap in zip(evt_numbers, pmaps.values()):
        for peak_number, peak in enumerate(getattr(pmap, attr)):
            size  = peak.times    .size
            npmt  = peak.pmts .ids.size
            nsipm = peak.sipms.ids.size

            data["evt_numbers"]      .extend([ evt_number] * size)
            data["peak_numbers"]     .extend([peak_number] * size)
            data["evt_numbers_pmt"]  .extend([ evt_number] * size * npmt )
            data["peak_numbers_pmt"] .extend([peak_number] * size * npmt )
            data["evt_numbers_sipm"] .extend([ evt_number] * size * nsipm)
            data["peak_numbers_sipm"].extend([peak_number] * size * nsipm)
            data["times"]            .extend(peak.times)
            data["bwidths"]          .extend(peak.bin_widths)
            data["npmts"]            .extend(np.repeat(peak. pmts.ids, size))
            data["nsipms"]           .extend(np.repeat(peak.sipms.ids, size))
            data["enes"]             .extend(peak.pmts.sum_over_sensors)
            data["enes_pmt"]         .extend(peak.pmts .all_waveforms.flatten())
            data["enes_sipm"]        .extend(peak.sipms.all_waveforms.flatten())

    data = {k: np.array(v) for k, v in data.items()}
    return pmaps_data(**data)


def test_make_tables(output_tmpdir):
    output_filename = os.path.join(output_tmpdir, "make_tables.h5")

    with tb.open_file(output_filename, "w") as h5f:
        pmpio._make_tables(h5f, None)

        assert "PMAPS" in h5f.root
        for tablename in ("S1", "S2", "S2Si", "S1Pmt", "S2Pmt"):
            assert tablename in h5f.root.PMAPS

            table = getattr(h5f.root.PMAPS, tablename)
            assert "columns_to_index" in table.attrs
            assert table.attrs.columns_to_index == ["event"]


def test_store_peak_s1(output_tmpdir, KrMC_pmaps_dict):
    output_filename = os.path.join(output_tmpdir, "store_peak_s1.h5")
    pmaps, _        = KrMC_pmaps_dict
    peak_number     = 0
    for evt_number, pmap in pmaps.items():
        if not pmap.s1s: continue

        with tb.open_file(output_filename, "w") as h5f:
            s1_table, _, _, s1i_table, _ = pmpio._make_tables(h5f, None)

            peak = pmap.s1s[peak_number]
            pmpio.store_peak(s1_table, s1i_table, None,
                             peak, peak_number, evt_number)
            h5f.flush()

            cols = h5f.root.PMAPS.S1.cols
            assert np.all(cols.event[:] == evt_number)
            assert np.all(cols.peak [:] == peak_number)
            assert        cols.time [:] == approx(peak.times)
            assert        cols.ene  [:] == approx(peak.pmts.sum_over_sensors)

            cols = h5f.root.PMAPS.S1Pmt.cols
            expected_npmt = np.repeat(peak.pmts.ids, peak.times.size)
            expected_enes = peak.pmts.all_waveforms.flatten()
            assert np.all(cols.event[:] == evt_number)
            assert np.all(cols.peak [:] == peak_number)
            assert        cols.npmt [:] == exactly(expected_npmt)
            assert        cols.ene  [:] == approx (expected_enes)


def test_store_peak_s2(output_tmpdir, KrMC_pmaps_dict):
    output_filename = os.path.join(output_tmpdir, "store_peak_s2.h5")
    pmaps, _        = KrMC_pmaps_dict
    peak_number     = 0
    for evt_number, pmap in pmaps.items():
        if not pmap.s2s: continue

        with tb.open_file(output_filename, "w") as h5f:
            _, s2_table, si_table, _, s2i_table = pmpio._make_tables(h5f, None)

            peak = pmap.s2s[peak_number]
            pmpio.store_peak(s2_table, s2i_table, si_table,
                             peak, peak_number, evt_number)
            h5f.flush()

            cols = h5f.root.PMAPS.S2.cols
            assert np.all(cols.event[:] == evt_number)
            assert np.all(cols.peak [:] == peak_number)
            assert        cols.time [:] == approx(peak.times)
            assert        cols.ene  [:] == approx(peak.pmts.sum_over_sensors)

            cols = h5f.root.PMAPS.S2Pmt.cols
            expected_npmt = np.repeat(peak.pmts.ids, peak.times.size)
            expected_enes = peak.pmts.all_waveforms.flatten()
            assert np.all(cols.event[:] == evt_number)
            assert np.all(cols.peak [:] == peak_number)
            assert        cols.npmt [:] == exactly(expected_npmt)
            assert        cols.ene  [:] == approx (expected_enes)

            cols = h5f.root.PMAPS.S2Si.cols
            expected_nsipm = np.repeat(peak.sipms.ids, peak.times.size)
            expected_enes  = peak.sipms.all_waveforms.flatten()
            assert np.all(cols.event[:] == evt_number)
            assert np.all(cols.peak [:] == peak_number)
            assert        cols.nsipm[:] == exactly(expected_nsipm)
            assert        cols.ene  [:] == approx (expected_enes)


def test_store_pmap(output_tmpdir, KrMC_pmaps_dict):
    output_filename = os.path.join(output_tmpdir, "store_pmap.h5")
    pmaps, _        = KrMC_pmaps_dict
    evt_numbers_set = np.random.randint(100, 200, size=len(pmaps))
    with tb.open_file(output_filename, "w") as h5f:
        tables = pmpio._make_tables(h5f, None)
        for evt_number, pmap in zip(evt_numbers_set, pmaps.values()):
            pmpio.store_pmap(tables, pmap, evt_number)
        h5f.flush()

        s1_data = pmaps_to_arrays(pmaps, evt_numbers_set, "s1s")

        cols = h5f.root.PMAPS.S1.cols
        assert cols.event[:] == exactly(s1_data. evt_numbers)
        assert cols.peak [:] == exactly(s1_data.peak_numbers)
        assert cols.time [:] == approx (s1_data.times)
        assert cols.ene  [:] == approx (s1_data.enes)


        cols = h5f.root.PMAPS.S1Pmt.cols
        assert cols.event[:] == exactly(s1_data.evt_numbers_pmt)
        assert cols.peak [:] == exactly(s1_data.peak_numbers_pmt)
        assert cols.npmt [:] == approx (s1_data.npmts)
        assert cols.ene  [:] == approx (s1_data.enes_pmt)

        s2_data = pmaps_to_arrays(pmaps, evt_numbers_set, "s2s")

        cols = h5f.root.PMAPS.S2.cols
        assert cols.event[:] == exactly(s2_data.evt_numbers)
        assert cols.peak [:] == exactly(s2_data.peak_numbers)
        assert cols.time [:] == approx (s2_data.times)
        assert cols.ene  [:] == approx (s2_data.enes)

        cols = h5f.root.PMAPS.S2Pmt.cols
        assert cols.event[:] == exactly(s2_data.evt_numbers_pmt)
        assert cols.peak [:] == exactly(s2_data.peak_numbers_pmt)
        assert cols.npmt [:] == exactly(s2_data.npmts)
        assert cols.ene  [:] == approx (s2_data.enes_pmt)

        cols = h5f.root.PMAPS.S2Si.cols
        assert cols.event[:] == exactly(s2_data.evt_numbers_sipm)
        assert cols.peak [:] == exactly(s2_data.peak_numbers_sipm)
        assert cols.nsipm[:] == exactly(s2_data.nsipms)
        assert cols.ene  [:] == approx (s2_data.enes_sipm)


def test_check_file_integrity_ok(KrMC_pmaps_filename):
    """For a file with no problems (like the input here), this test should pass."""
    # just check that it doesn't raise an exception
    with tb.open_file(KrMC_pmaps_filename) as file:
        pmpio.check_file_integrity(file)


def test_check_file_integrity_raises(KrMC_pmaps_filename, config_tmpdir):
    """Check that a file with a mismatch in the event number raises an exception."""
    filename = os.path.join(config_tmpdir, f"test_check_file_integrity_raises.h5")
    shutil.copy(KrMC_pmaps_filename, filename)
    with tb.open_file(filename, "r+") as file:
        file.root.Run.events.remove_rows(0, 1) # remove first row/event

        with raises(InvalidInputFileStructure, match="Inconsistent data: event number mismatch"):
            pmpio.check_file_integrity(file)


def test_load_pmaps_as_df_eager(two_pmaps):
    filename, _, true_dfs = two_pmaps
    read_dfs = pmpio.load_pmaps_as_df_eager(filename)
    for read_df, true_df in zip(read_dfs, true_dfs):
        assert_dataframes_equal(read_df, true_df)


def test_load_pmaps_as_df_lazy(KrMC_pmaps_filename):
    """Ensure the lazy and non-lazy versions provide the same result"""
    dfs_eager = pmpio.load_pmaps_as_df_eager(KrMC_pmaps_filename)
    dfs_lazy  = pmpio.load_pmaps_as_df_lazy (KrMC_pmaps_filename)

    # concatenate all dfs from the same node
    dfs_lazy = [pd.concat(node_dfs, ignore_index=True) for node_dfs in zip(*dfs_lazy)]

    assert len(dfs_eager) == len(dfs_lazy)
    for df_lazy, df_eager in zip(dfs_lazy, dfs_eager):
        assert_dataframes_equal(df_lazy, df_eager)


@mark.parametrize("skip" , (0, 1, 2, 3))
@mark.parametrize("nread", (0, 1, 2, 3))
def test_load_pmaps_as_df_lazy_subset(KrMC_pmaps_filename, skip, nread):
    """Ensure the reader provides the expected number of events"""
    # concatenate all dfs from the same node
    dfs_lazy  = pmpio.load_pmaps_as_df_lazy (KrMC_pmaps_filename, skip, nread)
    dfs_lazy = [pd.concat(node_dfs, ignore_index=True) for node_dfs in zip(*dfs_lazy)]

    for df in dfs_lazy:
        assert df.event.nunique() == nread


def test_load_pmaps_as_df(KrMC_pmaps_filename):
    """Ensure the output of the function is the expected one"""
    eager = pmpio.load_pmaps_as_df(KrMC_pmaps_filename, lazy=False)
    lazy  = pmpio.load_pmaps_as_df(KrMC_pmaps_filename, lazy=True )
    assert len(eager) == 5
    assert all(isinstance(item, pd.DataFrame) for item in eager)

    assert isinstance(lazy, Generator)
    element = next(lazy)
    assert len(element) == 5
    assert all(isinstance(item, pd.DataFrame) for item in element)


@mark.skip(reason="Deprecated feature. Plus this test makes no sense. Compares the output with itself.")
def test_load_pmaps_as_df_eager_without_ipmt(KrMC_pmaps_without_ipmt_filename, KrMC_pmaps_without_ipmt_dfs):
    true_dfs = KrMC_pmaps_without_ipmt_dfs
    read_dfs = pmpio.load_pmaps_as_df_eager(KrMC_pmaps_without_ipmt_filename)

    # Indices 0, 1 and 2 correspond to the S1sum, S2sum and Si
    # dataframes. Indices 3 and 4 are the S1pmt and S2pmt dataframes
    # which should be None when not present.
    for read_df, true_df in zip(read_dfs[:3], true_dfs[:3]):
        assert_dataframes_equal(read_df, true_df)

    assert read_dfs[3] is None
    assert read_dfs[4] is None


def test_load_pmaps_eager(two_pmaps, output_tmpdir):
    filename, true_pmaps, _ = two_pmaps
    read_pmaps = pmpio.load_pmaps_eager(filename)
    assert    len(read_pmaps) ==    len(true_pmaps)
    assert sorted(read_pmaps) == sorted(true_pmaps) # compare keys
    for key, true_pmap in true_pmaps.items():
        assert_PMap_equality(read_pmaps[key], true_pmap)


def test_load_pmaps_lazy(KrMC_pmaps_filename):
    """Ensure the lazy and non-lazy versions provide the same result"""
    pmaps_eager = pmpio.load_pmaps_eager(KrMC_pmaps_filename)
    pmaps_lazy  = pmpio.load_pmaps_lazy (KrMC_pmaps_filename)

    # combine all events in the same dictionary
    pmaps_lazy = dict(pmaps_lazy)

    assert len(pmaps_lazy) == len(pmaps_eager)
    for evt, pmap_lazy in pmaps_lazy.items():
        assert evt in pmaps_eager
        assert_PMap_equality(pmap_lazy, pmaps_eager[evt])


@mark.parametrize("skip" , (0, 1, 2, 3))
@mark.parametrize("nread", (0, 1, 2, 3))
def test_load_pmaps_lazy_subset(KrMC_pmaps_filename, skip, nread):
    """Ensure the reader provides the expected number of events"""
    # concatenate all dfs from the same node
    pmaps_lazy  = pmpio.load_pmaps_lazy(KrMC_pmaps_filename, skip, nread)
    pmaps_lazy = dict(pmaps_lazy)
    assert len(pmaps_lazy) == nread


def test_load_pmaps(KrMC_pmaps_filename):
    """Ensure the output of the function is the expected one"""
    eager = pmpio.load_pmaps(KrMC_pmaps_filename, lazy=False)
    lazy  = pmpio.load_pmaps(KrMC_pmaps_filename, lazy=True )

    assert isinstance(eager, Mapping)
    element = next(iter(eager.values()))
    assert isinstance(element, PMap)

    assert isinstance(lazy, Generator)
    element = next(lazy)
    assert len(element) == 2 # event, pmap
    assert isinstance(element[1], PMap)


@mark.parametrize("signal_type", (S1, S2))
def test_build_pmt_responses(KrMC_pmaps_dfs, signal_type):
    if   signal_type is S1:   df,  _, _, pmt_df,      _ = KrMC_pmaps_dfs
    elif signal_type is S2:    _, df, _,      _, pmt_df = KrMC_pmaps_dfs

    df_groupby     =     df.groupby(["event", "peak"])
    pmt_df_groupby = pmt_df.groupby(["event", "peak"])
    for (_, df_peak), (_, pmt_df_peak) in zip(df_groupby, pmt_df_groupby):
        times, widths, pmt_r = pmpio.build_pmt_responses(df_peak, pmt_df_peak)

        assert times                         == approx(    df_peak.time.values)
        assert pmt_r.sum_over_sensors        == approx(    df_peak.ene .values)
        assert pmt_r.all_waveforms.flatten() == approx(pmt_df_peak.ene .values)


def test_build_sipm_responses(KrMC_pmaps_dfs):
    _, _, df_event, _, _ = KrMC_pmaps_dfs
    for _, df_peak in df_event.groupby(["event", "peak"]):
        sipm_r = pmpio.build_sipm_responses(df_peak)

        assert sipm_r.all_waveforms.flatten() == approx(df_peak.ene.values)


def test_build_pmt_responses_unordered():
    data_ipmt = pd.DataFrame(dict(
    event = np.zeros(10),
    peak  = np.zeros(10),
    npmt  = np.array([1, 1, 3, 3, 2, 2, 0, 0, 4, 4]),
    ene   = np.arange(10)))

    data_pmt = pd.DataFrame(dict(
    event = np.zeros (2),
    peak  = np.zeros (2),
    time  = np.arange(2),
    ene   = np.array ([20, 25])))


    expected_pmts = np.array([1, 3, 2, 0, 4])
    expected_enes = np.arange(10)
    for (_, peak_pmt), (_, peak_ipmt) in zip(data_pmt .groupby(["event", "peak"]),
                                             data_ipmt.groupby(["event", "peak"])):
        _, _, pmt_r = pmpio.build_pmt_responses(peak_pmt, peak_ipmt)
        assert pmt_r.ids                     == exactly(expected_pmts)
        assert pmt_r.all_waveforms.flatten() == exactly(expected_enes)


def test_build_sipm_responses_unordered():
    data = pd.DataFrame(dict(
    event = np.zeros(10),
    peak  = np.zeros(10),
    nsipm = np.array([1, 1, 3, 3, 2, 2, 0, 0, 4, 4]),
    ene   = np.arange(10)))

    expected_sipms = np.array([1, 3, 2, 0, 4])
    expected_enes  = np.arange(10)
    for _, peak in data.groupby(["event", "peak"]):
        sipm_r = pmpio.build_sipm_responses(peak)
        assert sipm_r.ids                     == exactly(expected_sipms)
        assert sipm_r.all_waveforms.flatten() == exactly(expected_enes)
