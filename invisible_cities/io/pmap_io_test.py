"""
code: pmap_io_test.py
"""
import os
import time

import tables as tb
import numpy  as np

from pytest import mark

from hypothesis             import given
from hypothesis.strategies  import floats
from hypothesis.strategies  import integers

from .. core.system_of_units_c import units
from .. database               import load_db
from .. sierpe                 import blr

from .. reco                   import tbl_functions    as tbl
from .. reco                   import peak_functions   as pf
from .. reco                   import peak_functions_c as cpf

from .. reco.params            import S12Params        as S12P
from .. reco.params            import ThresholdParams
from .. reco.params            import PMaps
from .. types.ic_types         import minmax

from . pmap_io                 import pmap_writer
from .. evm.pmaps              import S1
from .. evm.pmaps              import S2
from .. evm.pmaps              import S2Si
from . run_and_event_io        import run_and_event_writer

from . pmap_io                 import load_pmaps
from . pmap_io                 import s1_s2_si_from_pmaps
from . pmap_io                 import read_run_and_event_from_pmaps_file
from . pmap_io   import df_to_s1_dict
from . pmap_io   import df_to_s2_dict
from . pmap_io   import df_to_s2si_dict

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

