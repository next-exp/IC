from __future__ import absolute_import

import os
from operator import itemgetter
import numpy as np
from pandas import DataFrame as DF, Series
from pytest import fixture, mark

parametrize = mark.parametrize


import invisible_cities.reco.pmaps_functions   as pmapf
from   invisible_cities.reco.pmaps_functions import df_to_pmaps_dict, df_to_s2si_dict


@fixture(scope='module')
def KrMC_pmaps():
    # Input file was produced to contain exactly 15 S1 and 50 S2.
    test_file = os.path.expandvars("$ICDIR/database/test_data/KrMC_pmaps.h5")
    S1_evts   = [1, 2, 4, 6, 12, 14, 15, 16, 17, 18, 19, 21, 22, 23, 24]
    S2_evts   = list(range(50, 100))
    S2Si_evts = list(S2_evts)
    for i in [56, 60, 64, 68, 69, 71, 80, 83, 85, 94, 95]:
        S2Si_evts.pop(S2Si_evts.index(i))
    s1s, s2s, s2sis = pmapf.read_pmaps(test_file)
    return (s1s, s2s, s2sis), (S1_evts, S2_evts, S2Si_evts)


def test_df_to_pmaps_dict_limit_events(KrMC_pmaps):
    max_events = 10
    (s1s, s2s, s2sis), (S1_evts, _, _) = KrMC_pmaps
    s1dict = df_to_pmaps_dict(s1s, max_events)
    s2dict = df_to_pmaps_dict(s2s, max_events)
    assert sorted(s1dict.keys()) == S1_evts[:4]
    assert sorted(s2dict.keys()) == []


def test_df_to_pmaps_dict_take_all_events_if_limit_too_high(KrMC_pmaps):
    max_events_is_more_than_available = 10000
    (s1s, s2s, s2sis), (S1_evts, S2_evts, _) = KrMC_pmaps
    s1dict = df_to_pmaps_dict(s1s, max_events_is_more_than_available)
    s2dict = df_to_pmaps_dict(s2s, max_events_is_more_than_available)
    assert sorted(s1dict.keys()) == S1_evts
    assert sorted(s2dict.keys()) == S2_evts


def test_df_to_pmaps_dict_default_number_of_events(KrMC_pmaps):
    # Read all events
    (s1s, s2s, s2sis), (S1_evts, S2_evts, _) = KrMC_pmaps
    s1dict = df_to_pmaps_dict(s1s)
    s2dict = df_to_pmaps_dict(s2s)
    assert sorted(s1dict.keys()) == S1_evts
    assert sorted(s2dict.keys()) == S2_evts


def test_df_to_pmaps_dict_negative_limit_takes_all_events(KrMC_pmaps):
    # Read all events
    negative_max_events = -23
    (s1s, s2s, s2sis), (S1_evts, S2_evts, _) = KrMC_pmaps
    s1dict = df_to_pmaps_dict(s1s, negative_max_events)
    s2dict = df_to_pmaps_dict(s2s, negative_max_events)
    assert sorted(list(s1dict.keys())) == S1_evts
    assert sorted(list(s2dict.keys())) == S2_evts


@fixture(scope='module')
def s12_dataframe_converted(request):
    evs  = [   0,     0,     0,     0,     0,      3,     3]
    peak = [   0,     0,     1,     1,     1,      0,     0]
    time = [1000., 1025., 2050., 2075., 2100., 5000., 5025.]
    ene  = [123.4, 140.8, 730.0, 734.2, 732.7, 400.0, 420.3]
    df = DF.from_dict(dict(
        event  = Series(evs , dtype=np.int32),
        evtDaq = evs,
        peak   = Series(peak, dtype=np.int8),
        time   = Series(time, dtype=np.float32),
        ene    = Series(ene , dtype=np.float32),
    ))
    return df_to_pmaps_dict(df), df


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
###############################################################
def test_df_to_s2si_dict_limit_events(KrMC_pmaps):
    max_events = 10
    (_, _, s2sis), (_, _, S2Si_evts) = KrMC_pmaps
    S2Sidict = df_to_s2si_dict(s2sis, max_events)
    assert sorted(S2Sidict.keys()) == []



def test_df_to_s2si_dict_take_all_events_if_limit_too_high(KrMC_pmaps):
    max_events_is_more_than_available = 10000
    (_, _, s2sis), (_, _, S2Si_evts) = KrMC_pmaps
    S2Sidict = df_to_s2si_dict(s2sis, max_events_is_more_than_available)
    assert sorted(S2Sidict.keys()) == S2Si_evts


def test_df_to_s2si_dict_default_number_of_events(KrMC_pmaps):
    # Read all events
    (_, _, s2sis), (_, _, S2Si_evts) = KrMC_pmaps
    S2Sidict = df_to_s2si_dict(s2sis)
    assert sorted(S2Sidict.keys()) == S2Si_evts


def test_df_to_s2si_dict_negative_limit_takes_all_events(KrMC_pmaps):
    # Read all events
    negative_max_events = -23
    (_, _, s2sis), (_, _, S2Si_evts) = KrMC_pmaps
    S2Sidict = df_to_s2si_dict(s2sis, negative_max_events)
    assert sorted(S2Sidict.keys()) == S2Si_evts


@fixture(scope='module')
def s2si_dataframe_converted(request):
    evs  = [    0,     0,     0,     0,     0,     3,     3]
    peak = [    0,     0,     1,     1,     1,     0,     0]
    sipm = [    1,     2,     3,     4,     5,     5,     6]
    samp = [    0,     2,     0,     1,     2,     3,     4]
    ene  = [123.4, 140.8, 730.0, 734.2, 732.7, 400.0, 420.3]
    df = DF.from_dict(dict(
        event   = Series(evs , dtype=np.int32),
        evtDaq  = evs,
        peak    = Series(peak, dtype=np.int8),
        nsipm   = Series(sipm, dtype=np.int16),
        nsample = Series(samp, dtype=np.int16),
        ene     = Series(ene , dtype=np.float32),
    ))
    return df_to_s2si_dict(df), df


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
