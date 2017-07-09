from operator import itemgetter

import numpy as np


from pytest import mark
parametrize = mark.parametrize

from .. core           import system_of_units as units

from .                 import pmaps_functions as pmapf
#from . pmaps_functions_c import df_to_pmaps_dict
from . pmaps_functions_c import df_to_s1_dict
from . pmaps_functions_c import df_to_s2_dict
from . pmaps_functions_c import df_to_s2si_dict


def test_integrate_sipm_charges_in_peak_as_dict():
    sipm1 = 1000
    sipm2 = 1001
    qs1 = list(range(5))
    qs2 = list(range(10))
    sipms = {sipm1:     qs1,
             sipm2:     qs2 }
    Qs    = {sipm1: sum(qs1),
             sipm2: sum(qs2)}
    assert pmapf.integrate_sipm_charges_in_peak_as_dict(sipms) == Qs

def test_integrate_sipm_charges_in_peak():
    sipm1 = 1234
    sipm2 =  987
    qs1 = [8,6,9,3]
    qs2 = [4,1,9,6,7]
    sipms = {sipm1: qs1,
             sipm2: qs2}
    ids, Qs =  pmapf.integrate_sipm_charges_in_peak(sipms)
    assert np.array_equal(ids, np.array((  sipm1,    sipm2)))
    assert np.array_equal(Qs , np.array((sum(qs1), sum(qs2))))

def test_integrate_S2Si_charge():
    peak1 = 0
    sipm1_1, Q1_1 = 1000, list(range(5))
    sipm1_2, Q1_2 = 1001, list(range(10))

    peak2 = 1
    sipm2_1, Q2_1 =  999, [6,4,9]
    sipm2_2, Q2_2 =  456, [8,4,3,7,5]
    sipm2_3, Q2_3 = 1234, [6,2]
    sipm2_4, Q2_4 =  666, [0,0] # !!! Zero charge

    S2Si = {peak1 : {sipm1_1 : Q1_1,
                     sipm1_2 : Q1_2},
            peak2 : {sipm2_1 : Q2_1,
                     sipm2_2 : Q2_2,
                     sipm2_3 : Q2_3,
                     sipm2_4 : Q2_4}}

    integrated_S2Si = pmapf.integrate_S2Si_charge(S2Si)
    assert integrated_S2Si == {peak1 : {sipm1_1 : sum(Q1_1),
                                        sipm1_2 : sum(Q1_2)},
                               peak2 : {sipm2_1 : sum(Q2_1),
                                        sipm2_2 : sum(Q2_2),
                                        sipm2_3 : sum(Q2_3),
                                        sipm2_4 : sum(Q2_4)}}


def test_width():
    initial_time = 10000
    width = 1000
    times = range(10000,initial_time + width+1)
    # Convert times to ns
    times = list(map(lambda t: t * units.ns, times))
    assert width == pmapf.width(times)
    assert width * units.ns / units.mus == pmapf.width(times, to_mus=True)



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

    # S2s, S2Sis = s2, s2si
    # for S2, Si in zip(S2s.values(), S2Sis.values()):
    #     for p in S2:
    #         if p in Si:
    #             if len(Si[p]) > 0: # Not necessary once Irene run completely with fix
    #                 assert len(Si[p][next(iter(Si[p]))]) == len(S2[p][0])

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


# def test_load_pmaps(KrMC_pmaps):
#
#     _, (s1t, s2t, s2sit), (_, _, _), (s1, s2, s2si) = KrMC_pmaps
#     # S1D, S2D, S2SiD are {event:S1}, {event:S2}, {event:S2Si}
#     S1D              = df_to_s1_dict(s1t)
#     S2D              = df_to_s2_dict(s2t)
#     S2SiD            = df_to_s2si_dict(s2t, s2sit)
#
#     for event, ss1 in S1D.items():
#         for peak_no in ss1.peak_collection():
#             wf1 = ss1.peak_waveform(peak_no)
#             wf2 = s1.peak_waveform(peak_no)
#             assert np.allclose(wf1.t, wf2.t, rtol=1e-4)
#             assert np.allclose(wf1.E, wf2.E, rtol=1e-4)
        # for i, (t, E) in s1d.items():
        #     assert np.allclose(t, s1[event][i][0], rtol=1e-4)
        #     assert np.allclose(E, s1[event][i][1], rtol=1e-4)

    # for event, s2d in S2.items():
    #     for i, (t, E) in s2d.items():
    #         assert np.allclose(t, s2[event][i][0], rtol=1e-4)
    #         assert np.allclose(E, s2[event][i][1], rtol=1e-4)
    #
    # for event, s2sid in S2Si.items():
    #     for i, sipm in s2sid.items():
    #         for nsipm, E in sipm.items():
    #             assert np.allclose(E, s2si[event][i][nsipm], rtol=1e-4)



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

#
# def test_df_to_pmaps_dict_arrays_lengths_are_equal(KrMC_pmaps):
#     _, (_, s2s, _), (_, S2_evts, _), _  = KrMC_pmaps
#
#     s2dict = df_to_pmaps_dict(s2s)
#     for evt in s2dict.values():
#         for peak in evt.values():
#             assert len(peak.t) == len(peak.E)
#
#
# def test_df_to_pmaps_dict_one_entry_per_event(s12_dataframe_converted):
#     converted, original = s12_dataframe_converted
#     number_of_events_in_original = len(set(original.event))
#     assert len(converted) == number_of_events_in_original
#
#
# def test_df_to_pmaps_dict_event_numbers_should_be_keys(s12_dataframe_converted):
#     converted, original = s12_dataframe_converted
#     event_numbers_in_original = set(original.event)
#     for event_number in event_numbers_in_original:
#         assert event_number in converted
#
#
# def test_df_to_pmaps_dict_structure(s12_dataframe_converted):
#     converted, _ = s12_dataframe_converted
#     # Each event number is mapped to a subdict ...
#     for event_no, subdict in converted.items():
#         # in which peak numbers are mapped to a t,E pair ...
#         for peak_no, peak_data in subdict.items():
#             assert hasattr(peak_data, 't')
#             assert hasattr(peak_data, 'E')
#             for element in peak_data:
#                 assert type(element) is np.ndarray
#
#
# def test_df_to_pmaps_dict_events_contain_peaks(s12_dataframe_converted):
#     converted, _ = s12_dataframe_converted
#     # Multiple peaks in one event ...
#     # In event no 0, there are two peaks; evt 3 has one peak
#     assert len(converted[0]) == 2
#     assert len(converted[3]) == 1
#
#
# def test_df_to_pmaps_dict_events_data_correct(s12_dataframe_converted):
#     converted, original = s12_dataframe_converted
#
#     all_peaks = [ converted[event][peak]
#                   for event in sorted(converted)
#                   for peak  in sorted(converted[event])]
#
#     converted_time = np.concatenate(list(map(itemgetter(0), all_peaks)))
#     converted_ene  = np.concatenate(list(map(itemgetter(1), all_peaks)))
#
#     assert (converted_time == original.time).all()
#     assert (converted_ene  == original.ene ).all()
#
#
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
#

# def test_df_to_s2si_dict_number_of_slices_is_correct(KrMC_pmaps):
#     _, (_, s2s, s2sis), (_, S2_evts, _), _  = KrMC_pmaps
#     s2_dict   = df_to_s2_dict   (s2s)
#     s2si_dict = df_to_s2si_dict (s2s, s2sis)
#
#     event_numbers_seen_in_tracking_plane = set(s2_dict)
#     event_numbers_seen_in_energy_plane   = set(s2si_dict)
#
#     common_event_numbers = set.intersection(event_numbers_seen_in_energy_plane,
#                                             event_numbers_seen_in_tracking_plane)
#
#     for event_no in common_event_numbers:
#         s2si    = set(s2si_dict[event_no])
#         s2      = set(s2_dict  [event_no])
#
#         for peak_no in set.intersection(energy_peak_nos, tracking_peak_nos):
#             energy_peak   = S2_energy  [event_no][peak_no]
#             tracking_peak = S2_tracking[event_no][peak_no]
#
#             for sipm_no, tracking_peak_E in tracking_peak.items():
#                 assert len(energy_peak.E) == len(tracking_peak_E)
#
# #
# for event_no, s2 in s2_dict.items():
#     s2si = s2si_dict[event_no]
#     for peak_no in s2.peak_collection():
#         s2_ts = s2.peak_waveform(peak_no).number_of_samples
#         si_ts = s2si.peak_waveform(peak_no).number_of_samples
#         assert s2_ts == si_ts
#
#         for sipm_no in s2si.sipms_in_peak(peak_no):
#             sipm_ts = s2si.sipm_waveform(peak_no, sipm_no).number_of_samples
#             assert sipm_ts == s2_ts
# def test_df_to_s2si_dict_one_entry_per_event(s2si_dataframe_converted):
#     converted, original = s2si_dataframe_converted
#     number_of_events_in_original = len(set(original.event))
#     assert len(converted) == number_of_events_in_original
#
#
# def test_df_to_s2si_dict_event_numbers_should_be_keys(s2si_dataframe_converted):
#     converted, original = s2si_dataframe_converted
#     event_numbers_in_original = set(original.event)
#     for event_number in event_numbers_in_original:
#         assert event_number in converted
#
#
# def test_df_to_s2si_dict_structure(s2si_dataframe_converted):
#     converted, _ = s2si_dataframe_converted
#     # Each event number is mapped to a subdict ...
#     for event_no, subdict in converted.items():
#         # in which peak numbers are mapped to a subdict ...
#         for peak_no, peak_data in subdict.items():
#             # in which SiPMs IDs are mapped to  a t, E pair
#             for sipm_no, sipm in peak_data.items():
#                 assert type(sipm) is np.ndarray
#
#
# def test_df_to_s2si_dict_events_contain_peaks(s2si_dataframe_converted):
#     converted, _ = s2si_dataframe_converted
#     # Multiple peaks in one event ...
#     # In event no 0, there are two peaks; evt 3 has one peak
#     assert len(converted[0]) == 2
#     assert len(converted[3]) == 1
#
#
# def test_df_to_s2si_dict_events_data_correct(s2si_dataframe_converted):
#     converted, original = s2si_dataframe_converted
#
#     all_sipms = [ (sipm[0], sipm[1])
#                   for event in sorted(converted)
#                   for peak  in sorted(converted[event])
#                   for sipm  in sorted(converted[event][peak].items()) ]
#
#     converted_sipm = np.array      (list(map(itemgetter(0), all_sipms)))
#     converted_ene  = np.concatenate(list(map(itemgetter(1), all_sipms)))
#
#     assert (converted_sipm                            == original.nsipm).all()
#     assert (converted_ene[np.nonzero(converted_ene)]  == original.ene  ).all()
