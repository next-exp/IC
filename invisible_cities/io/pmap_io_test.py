"""
code: pmap_io_test.py
"""
import os
import time

import tables as tb
import numpy  as np

from pytest import mark

from hypothesis                import given
from hypothesis.strategies     import floats
from hypothesis.strategies     import integers

from .. core.system_of_units_c import units
from .. database               import load_db
from .. sierpe                 import blr

from .. reco                   import tbl_functions    as tbl
from .. reco                   import peak_functions   as pf
from .. reco                   import peak_functions_c as cpf

from .. evm.ic_containers      import S12Params        as S12P
from .. evm.ic_containers      import ThresholdParams
from .. evm.ic_containers      import PMaps
from .. types.ic_types         import minmax

from .. evm.pmaps              import S1
from .. evm.pmaps              import S2
from .. evm.pmaps              import S2Si
from .. evm.pmaps              import S1Pmt
from .. evm.pmaps              import S2Pmt
from . run_and_event_io        import run_and_event_writer

from . pmap_io                 import load_pmaps
from . pmap_io                 import load_ipmt_pmaps
from . pmap_io                 import load_pmaps_with_ipmt
from . pmap_io                 import pmap_writer
from . pmap_io                 import ipmt_pmap_writer
from . pmap_io                 import pmap_writer_and_ipmt_writer
from . pmap_io                 import s1_s2_si_from_pmaps
from . pmap_io                 import read_run_and_event_from_pmaps_file
from . pmap_io                 import df_to_s1_dict
from . pmap_io                 import df_to_s2_dict
from . pmap_io                 import df_to_s2si_dict

@given(integers(min_value=23, max_value=51))
def test_s1_s2_si_from_pmaps(KrMC_pmaps, evt_no):
    *_, (s1_dict,s2_dict, s2si_dict) = KrMC_pmaps

    s1, s2, s2si = s1_s2_si_from_pmaps(s1_dict, s2_dict, s2si_dict, evt_no)

    S1 = s1_dict  .get(evt_no, None)
    S2 = s2_dict  .get(evt_no, None)
    S2si = s2si_dict.get(evt_no, None)

    if S1 == None:
        assert S1 == s1
    else:
        assert s1.number_of_peaks == S1.number_of_peaks
        for i in s1.peak_collection():
            if s1.peak_waveform(i).good_waveform:
                np.allclose (s1.peak_waveform(i).t , S1.peak_waveform(i).t, rtol=1e-4)
                np.allclose (s1.peak_waveform(i).E , S1.peak_waveform(i).E, rtol=1e-4)

    if S2 == None:
        assert S2 == s2
    else:
        assert s2.number_of_peaks == S2.number_of_peaks
        for i in s2.peak_collection():
            if s2.peak_waveform(i).good_waveform:
                np.allclose (s2.peak_waveform(i).t , S2.peak_waveform(i).t, rtol=1e-4)
                np.allclose (s2.peak_waveform(i).E , S2.peak_waveform(i).E, rtol=1e-4)

    if S2si == None:
        assert S2si == s2si
    else:
        assert s2si.number_of_peaks == S2si.number_of_peaks
        for peak_number in s2si.peak_collection():
            assert (s2si.number_of_sipms_in_peak(peak_number) ==
                    S2si.number_of_sipms_in_peak(peak_number))
            for sipm_number in s2si.sipms_in_peak(peak_number):
                w   = s2si.sipm_waveform(peak_number, sipm_number)
                W   = S2si.sipm_waveform(peak_number, sipm_number)
                np.allclose(w.t , W.t)
                np.allclose(w.E , W.E)


def test_pmap_writer(config_tmpdir,
                     s1_dataframe_converted,
                     s2_dataframe_converted,
                     s2si_dataframe_converted):

    # setup temporary file
    filename = 'test_pmaps_auto.h5'
    PMP_file_name = os.path.join(config_tmpdir, filename)

    # Get test data
    s1_dict,   _ =  s1_dataframe_converted
    s2_dict,   _ =  s2_dataframe_converted
    s2si_dict, _ =  s2si_dataframe_converted

    #P = PMaps(s1_dict, s2_dict, s2si_dict)

    event_numbers = sorted(set(s1_dict).union(set(s2si_dict)))
    timestamps = { e : int(time.time() % 1 * 10 ** 9) for e in event_numbers }

    run_number = 632

    # Write pmaps to disk.
    with tb.open_file(PMP_file_name, 'w') as h5out:
        write_pmap          =          pmap_writer(h5out)
        write_run_and_event = run_and_event_writer(h5out)
        for event_no in event_numbers:
            timestamp = timestamps[event_no]
            s1        =   s1_dict[event_no]
            s2        =   s2_dict[event_no]
            s2si      = s2si_dict[event_no]

            write_pmap         (event_no, s1, s2, s2si)
            write_run_and_event(run_number, event_no, timestamp)

    # Read back the data we have just written
    S1D, S2D, S2SiD =  load_pmaps(PMP_file_name)
    rundf, evtdf = read_run_and_event_from_pmaps_file(PMP_file_name)

    # Convert them into our transient format
    # S1D   = df_to_s1_dict (s1df)
    # S2D   = df_to_s2_dict (s2df)
    # S2SiD = df_to_s2si_dict(s2df, s2sidf)

    ######################################################################
    # Compare original data to those read back

    for event_no, s1 in s1_dict.items():
        s1 = s1_dict[event_no]
        S1 = S1D[event_no]

        for peak_no in s1.peak_collection():
            PEAK = S1.peak_waveform(peak_no)
            peak = s1.peak_waveform(peak_no)
            np.testing.assert_allclose(peak.t, PEAK.t)
            np.testing.assert_allclose(peak.E, PEAK.E)

    for event_no, s2 in s2_dict.items():
        s2 = s2_dict[event_no]
        S2 = S2D[event_no]

        for peak_no in s2.peak_collection():
            PEAK = S2.peak_waveform(peak_no)
            peak = s2.peak_waveform(peak_no)
            np.testing.assert_allclose(peak.t, PEAK.t)
            np.testing.assert_allclose(peak.E, PEAK.E)

    for event_no, si in s2si_dict.items():
        si = s2si_dict[event_no]
        Si = S2SiD[event_no]

        for peak_no in si.peak_collection():
            PEAK = Si.peak_waveform(peak_no)
            peak = si.peak_waveform(peak_no)
            np.testing.assert_allclose(peak.t, PEAK.t)
            np.testing.assert_allclose(peak.E, PEAK.E)

            for sipm_no in si.sipms_in_peak(peak_no):
                sipm_wfm = si.sipm_waveform(peak_no, sipm_no)
                SIPM_wfm = Si.sipm_waveform(peak_no, sipm_no)
                np.testing.assert_allclose(sipm_wfm.t, SIPM_wfm.t)


    # Event numbers
    np.testing.assert_equal(evtdf.evt_number.values,
                            np.array(event_numbers, dtype=np.int32))

    # Run numbers
    np.testing.assert_equal(rundf.run_number.values,
                            np.full(len(event_numbers), run_number, dtype=np.int32))


def assert_s12_dict_equality(s12_dict0, s12_dict1):
    assert s12_dict0.keys() == s12_dict1.keys()
    for ev in s12_dict0:
        for pn in s12_dict0[ev].peaks:
            assert np.allclose(s12_dict0[ev].peaks[pn].t, s12_dict1[ev].peaks[pn].t)
            assert np.allclose(s12_dict0[ev].peaks[pn].E, s12_dict1[ev].peaks[pn].E)


def assert_s12pmt_dict_equality(s12pmt_dict0, s12pmt_dict1):
    assert_s12_dict_equality(s12pmt_dict0, s12pmt_dict1)
    for s12pmt0, s12pmt1 in zip(s12pmt_dict0.values(), s12pmt_dict1.values()):
        assert s12pmt0.ipmtd.keys() == s12pmt1.ipmtd.keys()
        for pn in s12pmt0.peaks:
            assert (s12pmt0.energies_in_peak(pn) == s12pmt1.energies_in_peak(pn)).all()


def assert_s2si_dict_equality(s2si_dict0, s2si_dict1):
    assert s2si_dict0.keys() == s2si_dict1.keys()
    for ev in s2si_dict0:
        for pn in s2si_dict0[ev].peaks:
            for sipm in s2si_dict0[ev].sipms_in_peak(pn):
                assert np.allclose(s2si_dict0[ev].sipm_waveform(pn, sipm).E,
                                   s2si_dict1[ev].sipm_waveform(pn, sipm).E)


def test_load_pmaps_with_ipmt_equality_with_load_pmaps_for_s1_s2_s2si(Kr_MC_4446_load_s1_s2_s2si,
                                                                 Kr_MC_4446_load_pmaps_with_ipmt):
    s1_dict0, s2_dict0, s2si_dict0       = Kr_MC_4446_load_s1_s2_s2si
    s1_dict1, s2_dict1, s2si_dict1, _, _ = Kr_MC_4446_load_pmaps_with_ipmt
    assert_s12_dict_equality (  s1_dict0,   s1_dict1) # check s1   equality
    assert_s12_dict_equality (  s2_dict0,   s2_dict1) # check s2   equality
    assert_s2si_dict_equality(s2si_dict0, s2si_dict1) # check s2si equality


def test_load_pmaps_with_ipmt_equality_with_load_ipmt_pmaps_for_ipmt(Kr_MC_4446_load_s1_s2_s2si,
                                                                Kr_MC_4446_load_pmaps_with_ipmt):
    test_ipmt_pmap_path = os.path.join(os.environ['ICDIR'],
                                       'database/test_data/Kr_MC_ipmt_pmaps_5evt.h5')
    s1pmt_dict0, s2pmt_dict0          = load_ipmt_pmaps(test_ipmt_pmap_path)
    _, _, _, s1pmt_dict1, s2pmt_dict1 = Kr_MC_4446_load_pmaps_with_ipmt
    assert_s12pmt_dict_equality (  s1pmt_dict0,   s1pmt_dict1)
    assert_s12pmt_dict_equality (  s2pmt_dict0,   s2pmt_dict1)


def test_load_ipmt_pmaps_s12pmt_have_same_events_and_peaks_as_s12(Kr_MC_4446_load_s1_s2_s2si,
                                                                  Kr_MC_4446_load_pmaps_with_ipmt):
    s1_dict, s2_dict, _             = Kr_MC_4446_load_s1_s2_s2si # loaded with load_pmaps
    _, _, _, s1pmt_dict, s2pmt_dict = Kr_MC_4446_load_pmaps_with_ipmt # loaded with load_ipmt_pmaps
    assert_s12_dict_equality(s1_dict, s1pmt_dict) # compares the s1.s1d with s1pmt.s1d
    assert_s12_dict_equality(s2_dict, s2pmt_dict) # compares the s2.s2d with s2pmt.s2d


def test_check_that_pmap_writer_and_ipmt_writer_does_not_modify_pmaps(config_tmpdir,
                                                                Kr_MC_4446_load_pmaps_with_ipmt):
    write_path = os.path.join(config_tmpdir, 'test_pmaps_auto.h5')
    s1_dict, s2_dict, s2si_dict, s1pmt_dict, s2pmt_dict = Kr_MC_4446_load_pmaps_with_ipmt
    with tb.open_file(write_path, 'w') as h5out:
        write_all_pmaps = pmap_writer_and_ipmt_writer(h5out)
        for ev in list(range(5)):
            if ev in    s1_dict:    s1 =    s1_dict[ev]  # Create empty si if there was no
            else:    s1 = None                           # peak of that kind found for that ev
            if ev in    s2_dict:    s2 =    s2_dict[ev]
            else:    s2 = None
            if ev in  s2si_dict:  s2si =  s2si_dict[ev]
            else:  s2si = None
            if ev in s1pmt_dict: s1pmt = s1pmt_dict[ev]
            else: s1pmt = None
            if ev in s2pmt_dict: s2pmt = s2pmt_dict[ev]
            else: s2pmt = None
            write_all_pmaps(ev, s1, s2, s2si, s1pmt, s2pmt)
        h5out.flush() # Flush cannot go inside writers since writers do not know
                      # how much they have written or how much they will write
    s1_dict0, s2_dict0, s2si_dict0, s1pmt_dict0, s2pmt_dict0 = load_pmaps_with_ipmt(write_path)
    assert_s12_dict_equality   (   s1_dict,    s1_dict0)
    assert_s12_dict_equality   (   s2_dict,    s2_dict0)
    assert_s12pmt_dict_equality(s1pmt_dict, s1pmt_dict0)
    assert_s12pmt_dict_equality(s2pmt_dict, s2pmt_dict0)
