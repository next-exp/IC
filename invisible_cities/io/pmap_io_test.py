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
from .. core.ic_types          import minmax

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



# @mark.slow
# def test_pmap_electrons_40keV(config_tmpdir):
#     # NB: avoid taking defaults for PATH_IN and PATH_OUT
#     # since they are in general test-specific
#     # NB: avoid taking defaults for run number (test-specific)
#
#     RWF_file_name  = os.path.join(os.environ['ICDIR'],
#               'database/test_data/',
#               'electrons_40keV_z250_RWF.h5')
#
#     PMAP_file_name = os.path.join(config_tmpdir,
#               'electrons_40keV_z250_PMP.h5')
#
#     s1_params = S12P(time   = minmax(min =     90 * units.mus,
#                                      max =    110 * units.mus),
#                      length = minmax(min =      4,
#                                      max =     20),
#                      stride              =      4,
#                      rebin               =  False)
#     s2_params = S12P(time =   minmax(min =    110 * units.mus,
#                                      max =   1190 * units.mus),
#                      length = minmax(min =     80,
#                                      max = 200000),
#                      stride              =     40,
#                      rebin               =   True)
#     thr = ThresholdParams(thr_s1   =  0.2 * units.pes,
#                           thr_s2   =  1   * units.pes,
#                           thr_MAU  =  3   * units.adc,
#                           thr_sipm =  5   * units.pes,
#                           thr_SIPM = 20   * units.pes)
#
#     run_number = 0
#
#     with tb.open_file(RWF_file_name,'r') as h5rwf:
#         with tb.open_file(PMAP_file_name, 'w') as h5out:
#             write_pmap          =          pmap_writer(h5out)
#             write_run_and_event = run_and_event_writer(h5out)
#             #waveforms
#             pmtrwf, pmtblr, sipmrwf = tbl.get_vectors(h5rwf)
#
#             # data base
#             DataPMT = load_db.DataPMT(run_number)
#             pmt_active = np.nonzero(DataPMT.Active.values)[0].tolist()
#             adc_to_pes = abs(DataPMT.adc_to_pes.values)
#             coeff_c = abs(DataPMT.coeff_c.values)
#             coeff_blr = abs(DataPMT.coeff_blr.values)
#             DataSiPM = load_db.DataSiPM()
#             adc_to_pes_sipm = DataSiPM.adc_to_pes.values
#
#             # number of events
#             NEVT = pmtrwf.shape[0]
#             # number of events for test (at most NEVT)
#             NTEST = 2
#             # loop
#             XS1L = []
#             XS2L = []
#             XS2SiL = []
#             for event in range(NTEST):
#                 # deconv
#                 CWF = blr.deconv_pmt(pmtrwf[event], coeff_c, coeff_blr, pmt_active)
#                 # calibrated sum
#                 csum, csum_mau = cpf.calibrated_pmt_sum(CWF,
#                                                     adc_to_pes,
#                                                     pmt_active = pmt_active,
#                                                     n_MAU=100,
#                                                     thr_MAU=thr.thr_MAU)
#                 # zs sum
#                 s2_ene, s2_indx = cpf.wfzs(csum, threshold=thr.thr_s2)
#                 s2_t = cpf._time_from_index(s2_indx)
#                 s1_ene, s1_indx = cpf.wfzs(csum_mau, threshold=thr.thr_s1)
#                 s1_t = cpf._time_from_index(s1_indx)
#
#                 # S1 and S2
#                 s1d = cpf.find_s12(csum, s1_indx, **s1_params._asdict())
#                 s2d = cpf.find_s12(csum, s2_indx, **s2_params._asdict())
#                 # sipm
#                 sipmzs = cpf.signal_sipm(sipmrwf[event], adc_to_pes_sipm,
#                                          thr=thr.thr_sipm, n_MAU=100)
#
#                 s2sid = cpf.sipm_s2_dict(sipmzs, s2d, thr = thr.thr_SIPM)
#
#                 # s1 = cpf.find_S12(csum, s1_indx, **s1_params._asdict())
#                 # s2 = cpf.find_S12(csum, s2_indx, **s2_params._asdict())
#                 # #S2Si
#                 #
#                 # SIPM = cpf.select_sipm(sipm)
#                 # s2si = pf.sipm_s2_dict(SIPM, s2, thr=thr.thr_SIPM)
#
#                 # tests:
#                 # energy vector and time vector equal in S1 and s2
#                 assert len(s1d[0][0]) == len(s1d[0][1])
#                 assert len(s2d[0][0]) == len(s2d[0][1])
#
#                 if s2d and s2sid:
#                     for nsipm in s2sid[0]:
#                         assert len(s2sid[0][nsipm]) == len(s2d[0][0])
#
#                 # make S1, S2 and S2Si objects (from dicts)
#                 S1 = S12(s1d)
#                 S2 = S12(s2d)
#                 Si = S2Si(s2sid)
#                 # store in lists for further testing
#                 XS1L.append(s1d)
#                 XS2L.append(s2d)
#                 XS2SiL.append(s2sid)
#                 # produce a fake timestamp (in real like comes from data)
#                 timestamp = int(time.time())
#                 # write to file
#                 write_pmap         (event, S1, S2, Si)
#                 write_run_and_event(run_number, event, timestamp)
#
#     # Read back
#     s1df, s2df, s2sidf = read_pmaps(PMAP_file_name)
#     rundf, evtdf = read_run_and_event_from_pmaps_file(PMAP_file_name)
#     # get the dicts
#
#     S1L = df_to_pmaps_dict(s1df)
#     S2L = df_to_pmaps_dict(s2df)
#     S2SiL = df_to_s2si_dict(s2sidf)
#
#     #test
#     for event in range(len(XS1L)):
#         s1 = XS1L[event]
#         if s1: # dictionary not empty
#             s1p = S1L[event]
#             for peak_number in s1p:
#                 np.testing.assert_allclose(s1p[peak_number].t,
#                                            s1[peak_number][0])
#                 np.testing.assert_allclose(s1p[peak_number].E,
#                                            s1[peak_number][1])
#         s2 = XS2L[event]
#         if s2:
#             s2p = S2L[event]
#             for peak_number in s2p:
#                 np.testing.assert_allclose(s2p[peak_number].t,
#                                        s2[peak_number][0])
#                 np.testing.assert_allclose(s2p[peak_number].E,
#                                        s2[peak_number][1])
#         s2si = XS2SiL[event]
#         if s2si:
#             sip = S2SiL[event]
#             for peak_number in sip:
#                 sipm = sip[peak_number]
#                 sipm2 = s2si[peak_number]
#                 for nsipm in sipm:
#                     np.testing.assert_allclose(sipm[nsipm], sipm2[nsipm])
