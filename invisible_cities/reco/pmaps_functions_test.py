from __future__ import absolute_import

import os
from pytest import fixture

import invisible_cities.reco.pmaps_functions as pmapf

@fixture(scope='module')
def KrMC_pmaps():
    # Input file was produced to contain exactly 14 S1 and 107 S2.
    NS1, NS2 = 14, 107
    return (os.path.expandvars("$ICDIR/database/test_data/KrMC_pmaps.h5"),
            NS1, NS2)

def test_read_max_events(KrMC_pmaps):
    # Read a max of 10 events (NS1=4, NS2=10)
    pmaps_file, _, _ = KrMC_pmaps
    s1s, s2s, s2sis = pmapf.read_pmaps(pmaps_file)
    assert len(pmapf.s12df_to_s12l(s1s, 10)) == 4
    assert len(pmapf.s12df_to_s12l(s2s, 10)) == 10

def test_read_all_available_events_when_too_many_required(KrMC_pmaps):
    # Read a maximum of 10000 events
    pmaps_file, NS1, NS2 = KrMC_pmaps
    s1s, s2s, s2sis = pmapf.read_pmaps(pmaps_file)
    assert len(pmapf.s12df_to_s12l(s1s, 10000)) == NS1
    assert len(pmapf.s12df_to_s12l(s2s, 10000)) == NS2

def test_read_default_number_of_events(KrMC_pmaps):
    # Read all events
    pmaps_file, NS1, NS2 = KrMC_pmaps
    s1s, s2s, s2sis = pmapf.read_pmaps(pmaps_file)
    assert len(pmapf.s12df_to_s12l(s1s)) == NS1
    assert len(pmapf.s12df_to_s12l(s2s)) == NS2

def test_read_with_negative_default(KrMC_pmaps):
    # Read all events
    pmaps_file, NS1, NS2 = KrMC_pmaps
    s1s, s2s, s2sis = pmapf.read_pmaps(pmaps_file)
    assert len(pmapf.s12df_to_s12l(s1s, -23)) == NS1
    assert len(pmapf.s12df_to_s12l(s2s, -23)) == NS2
