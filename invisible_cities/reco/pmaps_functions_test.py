from operator import itemgetter

import numpy as np


from pytest import mark
parametrize = mark.parametrize

from .. core           import system_of_units as units
from . pmaps_functions_c import df_to_s1_dict
from . pmaps_functions_c import df_to_s2_dict
from . pmaps_functions_c import df_to_s2si_dict

def test_rebin_s2_yeilds_output_of_correct_len(KrMC_pmaps):
    _, (_, _, _), (_, _, _), (_, s2_dict, s2si_dict) = KrMC_pmaps


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
def test_df_to_pmaps_dict_limit_events(KrMC_pmaps):
    max_events = 30
    _, (s1t, s2t, s2sit), (S1_evts, _, _), _ = KrMC_pmaps
    s1dict = df_to_s1_dict(s1t, max_events)
    s2dict = df_to_s2_dict(s2t, max_events)
    assert sorted(s1dict.keys()) == S1_evts[:7]
    assert sorted(s2dict.keys()) == []

#
def test_df_to_pmaps_dict_take_all_events_if_limit_too_high(KrMC_pmaps):
    max_events_is_more_than_available = 10000
    _, (s1s, s2s, s2sis), (S1_evts, S2_evts, _), _  = KrMC_pmaps
    s1dict = df_to_s1_dict(s1s, max_events_is_more_than_available)
    s2dict = df_to_s2_dict(s2s, max_events_is_more_than_available)
    assert sorted(s1dict.keys()) == S1_evts
    assert sorted(s2dict.keys()) == S2_evts

#
def test_df_to_pmaps_dict_default_number_of_events(KrMC_pmaps):
    # Read all events
    _, (s1s, s2s, s2sis), (S1_evts, S2_evts, _), _  = KrMC_pmaps
    s1dict = df_to_s1_dict(s1s)
    s2dict = df_to_s2_dict(s2s)
    assert sorted(s1dict.keys()) == S1_evts
    assert sorted(s2dict.keys()) == S2_evts

#
def test_df_to_pmaps_dict_negative_limit_takes_all_events(KrMC_pmaps):
    # Read all events
    negative_max_events = -23
    _, (s1s, s2s, s2sis), (S1_evts, S2_evts, _), _  = KrMC_pmaps
    s1dict = df_to_s1_dict(s1s, negative_max_events)
    s2dict = df_to_s2_dict(s2s, negative_max_events)
    assert sorted(list(s1dict.keys())) == S1_evts
    assert sorted(list(s2dict.keys())) == S2_evts


# ###############################################################
# # df_to_s2si_dict-related tests
# ###############################################################
def test_df_to_s2si_dict_limit_events(KrMC_pmaps):
    max_events = 30
    _, (_, s2t, s2sit), (_, _, S2Si_evts), _  = KrMC_pmaps
    s2si_dict = df_to_s2si_dict(s2t, s2sit, max_events)
    assert sorted(s2si_dict.keys()) == []

#
def test_df_to_s2si_dict_take_all_events_if_limit_too_high(KrMC_pmaps):
    max_events_is_more_than_available = 10000
    _, (_, s2t, s2sit), (_, _, S2Si_evts), _  = KrMC_pmaps
    s2si_dict = df_to_s2si_dict(s2t, s2sit, max_events_is_more_than_available)
    assert sorted(s2si_dict.keys()) == S2Si_evts

#
def test_df_to_s2si_dict_default_number_of_events(KrMC_pmaps):
    # Read all events
    _, (_, s2t, s2sit), (_, _, S2Si_evts), _  = KrMC_pmaps
    s2si_dict = df_to_s2si_dict(s2t, s2sit)
    assert sorted(s2si_dict.keys()) == S2Si_evts

#
def test_df_to_s2si_dict_negative_limit_takes_all_events(KrMC_pmaps):
    # Read all events
    negative_max_events = -23
    _, (_, s2t, s2sit), (_, _, S2Si_evts), _  = KrMC_pmaps
    s2si_dict = df_to_s2si_dict(s2t, s2sit, negative_max_events)
    assert sorted(s2si_dict.keys()) == S2Si_evts


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
