from operator import itemgetter

import numpy as np

from pytest import mark
parametrize = mark.parametrize

from .. evm.pmaps         import Peak
from .. evm.pmaps         import S2
from .. evm.pmaps         import S2Si
from .. evm.ic_containers import S12Params
from .. types.ic_types    import minmax
from .. core              import system_of_units as units
from . pmaps_functions    import rebin_s2si
from . pmaps_functions    import copy_s2si
from . pmaps_functions    import copy_s2si_dict
from . pmaps_functions    import raise_s2si_thresholds
from . pmaps_functions    import get_pmaps
from . pmaps_functions    import get_pmaps_with_ipmt
from . pmaps_functions_c  import _impose_thr_sipm_destructive
from . pmaps_functions_c  import _impose_thr_sipm_s2_destructive
from . pmaps_functions_c  import _delete_empty_s2si_peaks
from . pmaps_functions_c  import _delete_empty_s2si_dict_events
from . pmaps_functions_c  import df_to_s1_dict
from . pmaps_functions_c  import df_to_s2_dict
from . pmaps_functions_c  import df_to_s2si_dict
from . pmaps_functions_c  import sipm_ids_and_charges_in_slice


def test_equal_number_of_timebins_in_S2s_and_S2Sis(KrMC_pmaps):

    _, (_, _, _), (_, _, _), (_, s2_dict, s2si_dict) = KrMC_pmaps

    for event_no, s2 in s2_dict.items():
        s2si = s2si_dict[event_no]
        for peak_no in s2.peak_collection():
            s2_ts = s2.peak_waveform(peak_no).number_of_samples
            si_ts = s2si.peak_waveform(peak_no).number_of_samples
            assert s2_ts == si_ts

            for sipm_no in s2si.sipms_in_peak(peak_no):
                sipm_ts = s2si.sipm_waveform(peak_no, sipm_no).number_of_samples
                assert sipm_ts == s2_ts

def timebin_size_must_be_equal_to_stride_times_25_ns_shows_bug_in_old_data(KrMC_pmaps):
    """This fails when the time bin exceeds the allowed time range.
    This bug is an artifact of the old rebin_waveform function, and
    has been fixed now. Old data, like the one used in this test has the
    bug in it, and this test demonstrates it.
    """
    _, (_, _, _), (_, _, _), (_, s2, _) = KrMC_pmaps
    S2s = {0 : s2[31]}
    max_timebin_size = 1 * units.mus

    for S2s_ev in S2s.values(): # event loop
        for S2_p in S2s_ev.values(): # peak loop
            try:  # will fail, and we cath it and pass
                assert (np.array([S2_p[0][i] - S2_p[0][i-1] \
                        for i in range(1, len(S2_p[0]))])  \
                        <= max_timebin_size).all()
            except AssertionError:
                pass


###############################################################
# df_to_pmaps_dict-related tests
###############################################################

def test_df_to_s1s2si_dict_limit_events(KrMC_pmaps):
    _, (s1t, s2t, sit), (S1_evts, S2_evts, Si_evts), _ = KrMC_pmaps

    for i, max_events in enumerate(S1_evts):
        s1_dict = df_to_s1_dict(s1t, max_events)
        assert sorted(s1_dict.keys()) == S1_evts[:i]

    for i, max_events in enumerate(S2_evts):
        s2_dict = df_to_s2_dict(s2t, max_events)
        assert sorted(s2_dict.keys()) == S2_evts[:i]

    for i, max_events in enumerate(Si_evts):
        si_dict = df_to_s2si_dict(s2t, sit, max_events)
        assert sorted(si_dict.keys()) == Si_evts[:i]

#
def test_df_to_s1s2si_dict_take_all_events_if_limit_too_high(KrMC_pmaps):
    max_events_is_more_than_available = 10000
    _, (s1s, s2s, sis), (S1_evts, S2_evts, Si_evts), _  = KrMC_pmaps
    s1_dict = df_to_s1_dict  (s1s,      max_events_is_more_than_available)
    s2_dict = df_to_s2_dict  (s2s,      max_events_is_more_than_available)
    si_dict = df_to_s2si_dict(s2s, sis, max_events_is_more_than_available)

    assert sorted(s1_dict.keys()) == S1_evts
    assert sorted(s2_dict.keys()) == S2_evts
    assert sorted(si_dict.keys()) == Si_evts

#
def test_df_to_s1s2si_dict_default_number_of_events(KrMC_pmaps):
    # Read all events
    _, (s1s, s2s, sis), (S1_evts, S2_evts, Si_evts), _  = KrMC_pmaps
    s1_dict = df_to_s1_dict  (s1s)
    s2_dict = df_to_s2_dict  (s2s)
    si_dict = df_to_s2si_dict(s2s, sis)

    assert sorted(s1_dict.keys()) == S1_evts
    assert sorted(s2_dict.keys()) == S2_evts
    assert sorted(si_dict.keys()) == Si_evts

#
def test_df_to_s1s2si_dict_negative_limit_takes_all_events(KrMC_pmaps):
    # Read all events
    negative_max_events = -23
    _, (s1s, s2s, sis), (S1_evts, S2_evts, Si_evts), _  = KrMC_pmaps
    s1_dict = df_to_s1_dict  (s1s,      negative_max_events)
    s2_dict = df_to_s2_dict  (s2s,      negative_max_events)
    si_dict = df_to_s2si_dict(s2s, sis, negative_max_events)

    assert sorted(list(s1_dict.keys())) == S1_evts
    assert sorted(list(s2_dict.keys())) == S2_evts
    assert sorted(list(si_dict.keys())) == Si_evts


def test_df_to_s2si_dict_number_of_slices_is_correct(KrMC_pmaps):
    _, (_, s2s, s2sis), (_, S2_evts, _), _  = KrMC_pmaps
    s2_dict   = df_to_s2_dict   (s2s)
    s2si_dict = df_to_s2si_dict (s2s, s2sis)

    event_numbers_seen_in_tracking_plane = set(s2_dict)
    event_numbers_seen_in_energy_plane   = set(s2si_dict)

    common_event_numbers = set.intersection(event_numbers_seen_in_energy_plane,
                                            event_numbers_seen_in_tracking_plane)

    for event_no in common_event_numbers:
        s2si    = s2si_dict[event_no]
        s2      = s2_dict  [event_no]
        assert s2si.number_of_peaks == s2.number_of_peaks

        for peak_no in s2.peak_collection():
            s2_ts = s2.peak_waveform(peak_no).number_of_samples
            si_ts = s2si.peak_waveform(peak_no).number_of_samples
            assert s2_ts == si_ts

            for sipm_no in s2si.sipms_in_peak(peak_no):
                sipm_ts = s2si.sipm_waveform(peak_no, sipm_no).number_of_samples
                assert sipm_ts == s2_ts

# ###############################################################
# # rebin s2si-related tests
# ###############################################################
def test_rebinned_s2_energy_sum_same_as_original_energy_sum(KrMC_pmaps):
    _, (_, _, _), (_, _, _), (_, s2_dict, s2si_dict)  = KrMC_pmaps
    for s2, s2si in zip(s2_dict.values(), s2si_dict.values()):
        for rf in range(1,11):
            s2r, s2sir = rebin_s2si(s2, s2si, rf)
            for p in s2.s2d:
                assert s2.s2d[p][1].sum() == s2r.s2d[p][1].sum()
                for sipm in s2si.s2sid[p]:
                    assert s2si.s2sid[p][sipm].sum() == s2sir.s2sid[p][sipm].sum()


def test_rebinned_s2si_yeilds_correct_average_times(KrMC_pmaps):
    _, (_, _, _), (_, _, _), (_, s2_dict, s2si_dict)  = KrMC_pmaps
    for s2, s2si in zip(s2_dict.values(), s2si_dict.values()):
        for rf in range(1,11):
            s2r, s2sir = rebin_s2si(s2, s2si, rf)
            for p in s2.s2d:
                for i, t in enumerate(s2r.s2d[p][0]):
                    assert t == np.mean(s2.s2d[p][0][i*rf: min(i*rf + rf, len(s2.s2d[p][0]))])

#####

def test_sipm_ids_and_charges_in_slice(KrMC_pmaps):
    _, _, _, (_, _, s2si_dict) = KrMC_pmaps
    for s2si in s2si_dict.values():
        for s2sid_peak in s2si.s2sid.values():
            n_slices = len(s2sid_peak[list(s2sid_peak.keys())[0]])
            for i_slice in range(n_slices):
                ids, qs = sipm_ids_and_charges_in_slice(s2sid_peak, i_slice)
                for nsipm, q in zip(ids, qs):
                    assert s2sid_peak[nsipm][i_slice] == q


# ###############################################################
# raise s2si threshold related tests
# ###############################################################
def test_impose_thr_sipm_destructive_leaves_no_sipms_in_dict_with_lt_thr_sipm_charge(KrMC_pmaps):
    _, _, _, (_, _, s2si_dict0) = KrMC_pmaps
    thr_sipm = 20*units.pes
    s2si_dict = _impose_thr_sipm_destructive(s2si_dict0, thr_sipm)
    for ev in s2si_dict.keys():
        for pn in s2si_dict[ev].s2sid.keys():
            for qs in s2si_dict[ev].s2sid[pn].values():
                for q in qs:
                    if q != 0:
                        assert q > thr_sipm


def test_impose_thr_sipm_destructive_leaves_no_sipms_in_dict_with_0_integral_charge(KrMC_pmaps):
    _, _, _, (_, _, s2si_dict0) = KrMC_pmaps
    s2si_dict = _impose_thr_sipm_destructive(s2si_dict0, 20*units.pes)
    for ev in s2si_dict.keys():
        for pn in s2si_dict[ev].s2sid.keys():
            for sipm in s2si_dict[ev].s2sid[pn].keys():
                assert  s2si_dict[ev].s2sid[pn][sipm].sum() > 0


def test_impose_thr_sipm_destructive_does_does_nothing_with_smaller_threshold(KrMC_pmaps):
    _, _, _, (_, _, s2si_dict) = KrMC_pmaps
    s2si_dict1 = _impose_thr_sipm_destructive(s2si_dict, 0.01*units.pes)
    for ev in s2si_dict.keys():
        for pn in s2si_dict[ev].s2sid.keys():
            for sipm in s2si_dict[ev].s2sid[pn].keys():
                assert np.allclose(s2si_dict[ev].s2sid[pn][sipm], s2si_dict1[ev].s2sid[pn][sipm])


def test_impose_thr_sipm_s2_destructive_does_does_nothing_with_smaller_threshold(KrMC_pmaps):
    _, _, _, (_, _, s2si_dict) = KrMC_pmaps
    s2si_dict1 = _impose_thr_sipm_s2_destructive(s2si_dict, 0.01*units.pes)
    for ev in s2si_dict.keys():
        for pn in s2si_dict[ev].s2sid.keys():
            for sipm in s2si_dict[ev].s2sid[pn].keys():
                assert np.allclose(s2si_dict[ev].s2sid[pn][sipm], s2si_dict1[ev].s2sid[pn][sipm])


def test_impose_thr_sipm_s2_destructive_leaves_no_sipms_with_lt_thr_integral_charge(KrMC_pmaps):
    _, _, _, (_, _, s2si_dict) = KrMC_pmaps
    thr_sipm_s2 = 50*units.pes
    s2si_dict1 = _impose_thr_sipm_s2_destructive(s2si_dict, thr_sipm_s2)
    for s2si in s2si_dict1.values():
        for s2si_peak in s2si.s2sid.values():
            for qs in s2si_peak.values():
                assert qs.sum() > thr_sipm_s2


def test_delete_empty_s2si_peaks():
    s2d   = {0: (np.array([1, 2], dtype=np.float64), np.array([ 2,  2], dtype=np.float64)),
             1: (np.array([5, 6], dtype=np.float64), np.array([10, 10], dtype=np.float64))}
    s2sid = {0: {},
             1: {1000: np.array([1, 1], dtype=np.float64), 1001: np.array([3, 3], dtype=np.float64)}
             }
    s2si_dict = {0: S2Si(s2d, s2sid)}
    s2si_dict = _delete_empty_s2si_peaks(s2si_dict)
    assert len( s2si_dict[0].s2d    .keys())   == 1  # check s2d peak has been deleted
    assert len( s2si_dict[0].s2sid  .keys())   == 1  # check s2si peak has been deleted
    assert list(s2si_dict[0].peaks.keys())[0]  == 1  # check peak deleted in high level functions
    assert np.allclose(s2si_dict[0].s2sid[1][1000], s2sid[1][1000])
    assert np.allclose(s2si_dict[0].s2sid[1][1001], s2sid[1][1001])


def test_raise_s2si_thresholds_returns_empty_dict_with_enormous_thresholds(KrMC_pmaps):
    _, _, _, (_, _, s2si_dict) = KrMC_pmaps
    assert len(raise_s2si_thresholds(s2si_dict, None,  1e9)) == 0
    assert len(raise_s2si_thresholds(s2si_dict,  1e9, None)) == 0


def test_copy_s2si_changing_copy_does_not_affect_original():
    a     = np.array([[1,2],[1,2]], dtype=np.float64)
    b     = np.array([1,2], dtype=np.float64)
    s2d   = {0: np.array([[1,2],[1,2]], dtype=np.float64),
             2: np.array([[1,2],[1,2]], dtype=np.float64)}
    s2sid = {0: {1:b, 2: np.array([2,3], dtype=np.float64)},
             2: {     2: np.array([2,3], dtype=np.float64)}}
    s2si0 = S2Si(s2d, s2sid)
    s2si1 = copy_s2si(s2si0)
    # Check changing SiPM energy does not affect original
    s2si1.s2sid[0][2][0] -= 2
    assert s2si0.s2sid[0][2][0] == 2
    # Check deleting SiPM keys in copy does not affect original
    del s2si1.s2sid[0][1]
    assert (s2si0.s2sid[0][1] == b).all()
    # Check deleting peak keys in copy does not affect original in s2d
    del s2si1.s2sid[0]
    assert 0 in s2si0.s2sid
    # Check changing value in s2si1.s2d does not affect original
    s2si1.s2d[0][0][0] -= 1
    assert s2si0.s2d[0][0][0] == 1
    # Check deleting peak keys in copy does not affect original in s2sid
    del s2si1.s2d[2]
    assert 2 in s2si0.s2d


def test_copy_s2si_dict_deleting_keys_in_copy_does_not_affect_keys_in_original(KrMC_pmaps):
    _, _, _, (_, _, s2si_dict0) = KrMC_pmaps
    key  = list(s2si_dict0.keys())[-1]
    s2si_dict1 = copy_s2si_dict(s2si_dict0)
    del s2si_dict1[key]
    assert key in s2si_dict0


####
def test_get_pmap_functions_dont_crash_when_s2_is_None():
    index  = np.array(    [] , dtype=np.int32)
    csum   = np.ones(    100 , dtype=np.float64)
    sipmzs = np.ones(    100 , dtype=np.float64)
    ccwf   = np.ones((2, 100), dtype=np.float64) * 0.5
    params = S12Params(time=minmax(min=0.0, max=644000.0),
              stride=4,
              length=minmax(min=4, max=16),
              rebin_stride=1)
    thr_sipm_s2 = 5

    for s12 in get_pmaps(index, index, csum, sipmzs, params, params, thr_sipm_s2):
        assert s12 is None

    for s12 in get_pmaps_with_ipmt(index, index, ccwf, csum, sipmzs, params, params, thr_sipm_s2):
        assert s12 is None
