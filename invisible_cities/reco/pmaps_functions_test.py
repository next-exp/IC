from operator import itemgetter

import numpy as np


from pytest import mark
parametrize = mark.parametrize

from .. core           import system_of_units as units

from .                 import pmaps_functions as pmapf
from . pmaps_functions import df_to_pmaps_dict
from . pmaps_functions import df_to_s2si_dict


def test_integrate_charge():
    sipms = {1000: range(5),
             1001: range(10)}
    charges = np.array([[1000,1001],[10,45]])
    charges_test = pmapf.integrate_charge(sipms)
    np.testing.assert_array_equal(charges, charges_test)


def test_width():
    initial_time = 10000
    width = 1000
    times = range(10000,initial_time + width+1)
    # Convert times to ns
    times = list(map(lambda t: t * units.ns, times))
    assert width == pmapf.width(times)
    assert width * units.ns / units.mus == pmapf.width(times, to_mus=True)


def test_load_pmaps(KrMC_pmaps):

    _, (s1t, s2t, s2sit), (_, _, _), (s1, s2, s2si) = KrMC_pmaps
    S1              = df_to_pmaps_dict(s1t)
    S2              = df_to_pmaps_dict(s2t)
    S2Si            = df_to_s2si_dict(s2sit)

    for event, s1d in S1.items():
        for i, (t, E) in s1d.items():
            assert np.allclose(t, s1[event][i][0], rtol=1e-4)
            assert np.allclose(E, s1[event][i][1], rtol=1e-4)

    for event, s2d in S2.items():
        for i, (t, E) in s2d.items():
            assert np.allclose(t, s2[event][i][0], rtol=1e-4)
            assert np.allclose(E, s2[event][i][1], rtol=1e-4)

    for event, s2sid in S2Si.items():
        for i, sipm in s2sid.items():
            for nsipm, E in sipm.items():
                assert np.allclose(E, s2si[event][i][nsipm], rtol=1e-4)



###############################################################
# df_to_pmaps_dict-related tests
###############################################################
def test_df_to_pmaps_dict_limit_events(KrMC_pmaps):
    max_events = 30
    _, (s1s, s2s, s2sis), (S1_evts, _, _), _ = KrMC_pmaps
    s1dict = df_to_pmaps_dict(s1s, max_events)
    s2dict = df_to_pmaps_dict(s2s, max_events)
    assert sorted(s1dict.keys()) == S1_evts[:7]
    assert sorted(s2dict.keys()) == []


def test_df_to_pmaps_dict_take_all_events_if_limit_too_high(KrMC_pmaps):
    max_events_is_more_than_available = 10000
    _, (s1s, s2s, s2sis), (S1_evts, S2_evts, _), _  = KrMC_pmaps
    s1dict = df_to_pmaps_dict(s1s, max_events_is_more_than_available)
    s2dict = df_to_pmaps_dict(s2s, max_events_is_more_than_available)
    assert sorted(s1dict.keys()) == S1_evts
    assert sorted(s2dict.keys()) == S2_evts


def test_df_to_pmaps_dict_default_number_of_events(KrMC_pmaps):
    # Read all events
    _, (s1s, s2s, s2sis), (S1_evts, S2_evts, _), _  = KrMC_pmaps
    s1dict = df_to_pmaps_dict(s1s)
    s2dict = df_to_pmaps_dict(s2s)
    assert sorted(s1dict.keys()) == S1_evts
    assert sorted(s2dict.keys()) == S2_evts


def test_df_to_pmaps_dict_negative_limit_takes_all_events(KrMC_pmaps):
    # Read all events
    negative_max_events = -23
    _, (s1s, s2s, s2sis), (S1_evts, S2_evts, _), _  = KrMC_pmaps
    s1dict = df_to_pmaps_dict(s1s, negative_max_events)
    s2dict = df_to_pmaps_dict(s2s, negative_max_events)
    assert sorted(list(s1dict.keys())) == S1_evts
    assert sorted(list(s2dict.keys())) == S2_evts


def test_df_to_pmaps_dict_arrays_lengths_are_equal(KrMC_pmaps):
    _, (_, s2s, _), (_, S2_evts, _), _  = KrMC_pmaps

    s2dict = df_to_pmaps_dict(s2s)
    for evt in s2dict.values():
        for peak in evt.values():
            assert len(peak.t) == len(peak.E)


def test_df_to_pmaps_dict_one_entry_per_event(s12_dataframe_converted):
    converted, original = s12_dataframe_converted
    number_of_events_in_original = len(set(original.event))
    assert len(converted) == number_of_events_in_original


def test_df_to_pmaps_dict_event_numbers_should_be_keys(s12_dataframe_converted):
    converted, original = s12_dataframe_converted
    event_numbers_in_original = set(original.event)
    for event_number in event_numbers_in_original:
        assert event_number in converted


def test_df_to_pmaps_dict_structure(s12_dataframe_converted):
    converted, _ = s12_dataframe_converted
    # Each event number is mapped to a subdict ...
    for event_no, subdict in converted.items():
        # in which peak numbers are mapped to a t,E pair ...
        for peak_no, peak_data in subdict.items():
            assert hasattr(peak_data, 't')
            assert hasattr(peak_data, 'E')
            for element in peak_data:
                assert type(element) is np.ndarray


def test_df_to_pmaps_dict_events_contain_peaks(s12_dataframe_converted):
    converted, _ = s12_dataframe_converted
    # Multiple peaks in one event ...
    # In event no 0, there are two peaks; evt 3 has one peak
    assert len(converted[0]) == 2
    assert len(converted[3]) == 1


def test_df_to_pmaps_dict_events_data_correct(s12_dataframe_converted):
    converted, original = s12_dataframe_converted

    all_peaks = [ converted[event][peak]
                  for event in sorted(converted)
                  for peak  in sorted(converted[event])]

    converted_time = np.concatenate(list(map(itemgetter(0), all_peaks)))
    converted_ene  = np.concatenate(list(map(itemgetter(1), all_peaks)))

    assert (converted_time == original.time).all()
    assert (converted_ene  == original.ene ).all()


###############################################################
# df_to_s2si_dict-related tests
###############################################################
def test_df_to_s2si_dict_limit_events(KrMC_pmaps):
    max_events = 30
    _, (_, _, s2sis), (_, _, S2Si_evts), _  = KrMC_pmaps
    S2Sidict = df_to_s2si_dict(s2sis, max_events)
    assert sorted(S2Sidict.keys()) == []


def test_df_to_s2si_dict_take_all_events_if_limit_too_high(KrMC_pmaps):
    max_events_is_more_than_available = 10000
    _, (_, _, s2sis), (_, _, S2Si_evts), _  = KrMC_pmaps
    S2Sidict = df_to_s2si_dict(s2sis, max_events_is_more_than_available)
    assert sorted(S2Sidict.keys()) == S2Si_evts


def test_df_to_s2si_dict_default_number_of_events(KrMC_pmaps):
    # Read all events
    _, (_, _, s2sis), (_, _, S2Si_evts), _  = KrMC_pmaps
    S2Sidict = df_to_s2si_dict(s2sis)
    assert sorted(S2Sidict.keys()) == S2Si_evts


def test_df_to_s2si_dict_negative_limit_takes_all_events(KrMC_pmaps):
    # Read all events
    negative_max_events = -23
    _, (_, _, s2sis), (_, _, S2Si_evts), _  = KrMC_pmaps
    S2Sidict = df_to_s2si_dict(s2sis, negative_max_events)
    assert sorted(S2Sidict.keys()) == S2Si_evts


def test_df_to_s2si_dict_number_of_slices_is_correct(KrMC_pmaps):
    _, (_, s2s, s2sis), (_, S2_evts, _), _  = KrMC_pmaps
    S2_energy   = df_to_pmaps_dict(  s2s)
    S2_tracking = df_to_s2si_dict (s2sis)

    event_numbers_seen_in_tracking_plane = set(S2_tracking)
    event_numbers_seen_in_energy_plane   = set(S2_energy)

    common_event_numbers = set.intersection(event_numbers_seen_in_energy_plane,
                                            event_numbers_seen_in_tracking_plane)

    for event_no in common_event_numbers:
        tracking_peak_nos = set(S2_tracking[event_no])
        energy_peak_nos   = set(S2_energy  [event_no])

        for peak_no in set.intersection(energy_peak_nos, tracking_peak_nos):
            energy_peak   = S2_energy  [event_no][peak_no]
            tracking_peak = S2_tracking[event_no][peak_no]

            for sipm_no, tracking_peak_E in tracking_peak.items():
                assert len(energy_peak.E) == len(tracking_peak_E)


def test_df_to_s2si_dict_one_entry_per_event(s2si_dataframe_converted):
    converted, original = s2si_dataframe_converted
    number_of_events_in_original = len(set(original.event))
    assert len(converted) == number_of_events_in_original


def test_df_to_s2si_dict_event_numbers_should_be_keys(s2si_dataframe_converted):
    converted, original = s2si_dataframe_converted
    event_numbers_in_original = set(original.event)
    for event_number in event_numbers_in_original:
        assert event_number in converted


def test_df_to_s2si_dict_structure(s2si_dataframe_converted):
    converted, _ = s2si_dataframe_converted
    # Each event number is mapped to a subdict ...
    for event_no, subdict in converted.items():
        # in which peak numbers are mapped to a subdict ...
        for peak_no, peak_data in subdict.items():
            # in which SiPMs IDs are mapped to  a t, E pair
            for sipm_no, sipm in peak_data.items():
                assert type(sipm) is np.ndarray


def test_df_to_s2si_dict_events_contain_peaks(s2si_dataframe_converted):
    converted, _ = s2si_dataframe_converted
    # Multiple peaks in one event ...
    # In event no 0, there are two peaks; evt 3 has one peak
    assert len(converted[0]) == 2
    assert len(converted[3]) == 1


def test_df_to_s2si_dict_events_data_correct(s2si_dataframe_converted):
    converted, original = s2si_dataframe_converted

    all_sipms = [ (sipm[0], sipm[1])
                  for event in sorted(converted)
                  for peak  in sorted(converted[event])
                  for sipm  in sorted(converted[event][peak].items()) ]

    converted_sipm = np.array      (list(map(itemgetter(0), all_sipms)))
    converted_ene  = np.concatenate(list(map(itemgetter(1), all_sipms)))

    assert (converted_sipm                            == original.nsipm).all()
    assert (converted_ene[np.nonzero(converted_ene)]  == original.ene  ).all()
