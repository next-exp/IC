import os
from collections import namedtuple
from collections import defaultdict

from pytest import mark
from pytest import approx

import tables as tb
import numpy  as np

from ..core.testing_utils import assert_PMap_equality
from ..core.testing_utils import assert_dataframes_equal
from ..core.testing_utils import exactly
from ..evm .pmaps         import S1
from ..evm .pmaps         import S2
from .                    import pmaps_io as pmpio


pmaps_data = namedtuple("pmaps_data", """evt_numbers      peak_numbers
                                         evt_numbers_pmt  peak_numbers_pmt
                                         evt_numbers_sipm peak_numbers_sipm
                                         times npmts nsipms
                                         enes enes_pmt enes_sipm""")

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
        pmpio._make_tables(h5f)

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
            s1_table, _, _, s1i_table, _ = pmpio._make_tables(h5f)

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
            _, s2_table, si_table, _, s2i_table = pmpio._make_tables(h5f)

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
        tables = pmpio._make_tables(h5f)
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


def test_load_pmaps_as_df(KrMC_pmaps_filename, KrMC_pmaps_dfs):
    true_dfs = KrMC_pmaps_dfs
    read_dfs = pmpio.load_pmaps_as_df(KrMC_pmaps_filename)
    for read_df, true_df in zip(read_dfs, true_dfs):
        assert_dataframes_equal(read_df, true_df)


def test_load_pmaps_as_df_without_ipmt(KrMC_pmaps_without_ipmt_filename, KrMC_pmaps_without_ipmt_dfs):
    true_dfs = KrMC_pmaps_without_ipmt_dfs
    read_dfs = pmpio.load_pmaps_as_df(KrMC_pmaps_without_ipmt_filename)

    # Indices 0, 1 and 2 correspond to the S1sum, S2sum and Si
    # dataframes. Indices 3 and 4 are the S1pmt and S2pmt dataframes
    # which should be None when not present.
    for read_df, true_df in zip(read_dfs[:3], true_dfs[:3]):
        assert_dataframes_equal(read_df, true_df)

    assert read_dfs[3] is None
    assert read_dfs[4] is None


def test_load_pmaps(KrMC_pmaps_example):
    filename, true_pmaps = KrMC_pmaps_example
    read_pmaps = pmpio.load_pmaps(filename)

    assert len(read_pmaps)       == len(true_pmaps)
    assert     read_pmaps.keys() ==     true_pmaps.keys()
    for evt_number in true_pmaps:
        read_pmap = read_pmaps[evt_number]
        true_pmap = true_pmaps[evt_number]

        assert_PMap_equality(read_pmap, true_pmap)


@mark.parametrize("signal_type", (S1, S2))
def test_build_pmt_responses(KrMC_pmaps_dfs, signal_type):
    if   signal_type is S1:   df,  _, _, pmt_df,      _ = KrMC_pmaps_dfs
    elif signal_type is S2:    _, df, _,      _, pmt_df = KrMC_pmaps_dfs

    df_groupby     =     df.groupby(("event", "peak"))
    pmt_df_groupby = pmt_df.groupby(("event", "peak"))
    for (_, df_peak), (_, pmt_df_peak) in zip(df_groupby, pmt_df_groupby):
        times, pmt_r = pmpio.build_pmt_responses(df_peak, pmt_df_peak)

        assert times                         == approx(    df_peak.time.values)
        assert pmt_r.sum_over_sensors        == approx(    df_peak.ene .values)
        assert pmt_r.all_waveforms.flatten() == approx(pmt_df_peak.ene .values)


def test_build_sipm_responses(KrMC_pmaps_dfs):
    _, _, df_event, _, _ = KrMC_pmaps_dfs
    for _, df_peak in df_event.groupby(("event", "peak")):
        sipm_r = pmpio.build_sipm_responses(df_peak)

        assert sipm_r.all_waveforms.flatten() == approx(df_peak.ene.values)
