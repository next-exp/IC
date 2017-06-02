"""
code: pmap_io_test.py
"""
import os
import time

import tables as tb
import numpy  as np

from pytest import mark

from .. core.system_of_units_c import units
from .. database               import load_db
from .. sierpe                 import blr

from .                         import tbl_functions    as tbl
from .                         import peak_functions   as pf
from .                         import peak_functions_c as cpf

from . params                  import S12Params        as S12P
from . params                  import ThresholdParams
from . params                  import PMaps

from . pmap_io                 import pmap_writer
from .. reco.run_and_event_io  import run_and_event_writer
from . pmap_io                 import S12
from . pmap_io                 import S2Si

from . pmaps_functions         import read_pmaps
from . pmaps_functions         import read_run_and_event_from_pmaps_file
from . pmaps_functions_c       import df_to_pmaps_dict
from . pmaps_functions_c       import df_to_s2si_dict


def test_pmap_writer(config_tmpdir, s12_dataframe_converted, s2si_dataframe_converted):

    filename = 'test_pmaps_auto.h5'

    PMP_file_name = os.path.join(str(config_tmpdir), filename)

    # Get test data
    s12,  a =  s12_dataframe_converted
    s2si, b = s2si_dataframe_converted

    P = PMaps(s12, s12, s2si) # TODO Remove duplication of s12

    event_numbers = sorted(set(s12).union(set(s2si)))
    timestamps = { e : int(time.time() % 1 * 10 ** 9) for e in event_numbers }

    run_number = 632

    with tb.open_file(PMP_file_name, 'w') as h5out:
        # The actual pmap writing: the component whose functionality is
        # being tested here.
        write_pmap          =          pmap_writer(h5out)
        write_run_and_event = run_and_event_writer(h5out)
        for e in event_numbers:
            timestamp = timestamps[e]
            s1   = S12 (P.S1  .get(e, {}) )
            s2   = S12 (P.S2  .get(e, {}) )
            s2si = S2Si(P.S2Si.get(e, {}) )
            write_pmap         (e, s1, s2, s2si)
            write_run_and_event(run_number, e, timestamp)

    # Read back the data we have just written
    s1df, s2df, s2sidf =                   read_pmaps(PMP_file_name)
    rundf, evtdf = read_run_and_event_from_pmaps_file(PMP_file_name)

    # Convert them into our transient format
    S1D   = df_to_pmaps_dict (s1df)
    S2D   = df_to_pmaps_dict (s2df)
    S2SiD = df_to_s2si_dict(s2sidf)

    ######################################################################
    # Compare original data to those read back

    # The S12s
    for original_S, recovered_S in zip((  S1D,  S2D),
                                       (P.S1, P.S2)):
        for event_no, event in recovered_S.items():
            for peak_no, recovered_peak in event.items():
                original_peak = original_S[event_no][peak_no]
                np.testing.assert_allclose(recovered_peak.t, original_peak.t)
                np.testing.assert_allclose(recovered_peak.E, original_peak.E)

    # The S2Sis
    for event_no, event in S2SiD.items():
        for peak_no, peak in event.items():
            for sipm_id, recovered_Es in peak.items():
                original_Es = P.S2Si[event_no][peak_no][sipm_id]
                np.testing.assert_allclose(recovered_Es, original_Es)

    # Event numbers
    np.testing.assert_equal(evtdf.evt_number.values,
                            np.array(event_numbers, dtype=np.int32))

    # Run numbers
    np.testing.assert_equal(rundf.run_number.values,
                            np.full(len(event_numbers), run_number, dtype=np.int32))



@mark.slow
def test_pmap_electrons_40keV(config_tmpdir):
    # NB: avoid taking defaults for PATH_IN and PATH_OUT
    # since they are in general test-specific
    # NB: avoid taking defaults for run number (test-specific)

    RWF_file_name  = os.path.join(os.environ['ICDIR'],
              'database/test_data/',
              'electrons_40keV_z250_RWF.h5')

    PMAP_file_name = os.path.join(str(config_tmpdir),
              'electrons_40keV_z250_PMP.h5')

    s1_params = S12P(tmin=90*units.mus,
                 tmax=110*units.mus,
                 lmin=4,
                 lmax=20,
                 stride=4,
                 rebin=False)
    s2_params = S12P(tmin=110*units.mus,
                 tmax=1190*units.mus,
                 lmin=80,
                 lmax=200000,
                 stride=40,
                 rebin=True)
    thr = ThresholdParams(thr_s1=0.2*units.pes,
                          thr_s2=1*units.pes,
                          thr_MAU=3*units.adc,
                          thr_sipm=5*units.pes,
                          thr_SIPM=20*units.pes)

    run_number = 0

    with tb.open_file(RWF_file_name,'r') as h5rwf:
        with tb.open_file(PMAP_file_name, 'w') as h5out:
            write_pmap          =          pmap_writer(h5out)
            write_run_and_event = run_and_event_writer(h5out)
            #waveforms
            pmtrwf, pmtblr, sipmrwf = tbl.get_vectors(h5rwf)

            # data base
            DataPMT = load_db.DataPMT(run_number)
            pmt_active = np.nonzero(DataPMT.Active.values)[0].tolist()
            adc_to_pes = abs(DataPMT.adc_to_pes.values)
            coeff_c = abs(DataPMT.coeff_c.values)
            coeff_blr = abs(DataPMT.coeff_blr.values)
            DataSiPM = load_db.DataSiPM()
            adc_to_pes_sipm = DataSiPM.adc_to_pes.values

            # number of events
            NEVT = pmtrwf.shape[0]
            # number of events for test (at most NEVT)
            NTEST = 2
            # loop
            XS1L = []
            XS2L = []
            XS2SiL = []
            for event in range(NTEST):
                # deconv
                CWF = blr.deconv_pmt(pmtrwf[event], coeff_c, coeff_blr, pmt_active)
                # calibrated sum
                csum, csum_mau = cpf.calibrated_pmt_sum(CWF,
                                                    adc_to_pes,
                                                    pmt_active = pmt_active,
                                                    n_MAU=100,
                                                    thr_MAU=thr.thr_MAU)
                # zs sum
                s2_ene, s2_indx = cpf.wfzs(csum, threshold=thr.thr_s2)
                s2_t = cpf.time_from_index(s2_indx)
                s1_ene, s1_indx = cpf.wfzs(csum_mau, threshold=thr.thr_s1)
                s1_t = cpf.time_from_index(s1_indx)

                # S1 and S2
                s1 = cpf.find_S12(s1_ene, s1_indx, **s1_params._asdict())
                s2 = cpf.find_S12(s2_ene, s2_indx, **s2_params._asdict())
                #S2Si
                sipm = cpf.signal_sipm(sipmrwf[event],
                                       adc_to_pes_sipm,
                                       thr=thr.thr_sipm,
                                       n_MAU=100)
                SIPM = cpf.select_sipm(sipm)
                s2si = pf.sipm_s2_dict(SIPM, s2, thr=thr.thr_SIPM)

                # tests:
                # energy vector and time vector equal in S1 and s2
                assert len(s1[0][0]) == len(s1[0][1])
                assert len(s2[0][0]) == len(s2[0][1])

                if s2 and s2si:
                    for nsipm in s2si[0]:
                        assert len(s2si[0][nsipm]) == len(s2[0][0])

                # make S1, S2 and S2Si objects (from dicts)
                S1 = S12(s1)
                S2 = S12(s2)
                Si = S2Si(s2si)
                # store in lists for further testing
                XS1L.append(s1)
                XS2L.append(s2)
                XS2SiL.append(s2si)
                # produce a fake timestamp (in real like comes from data)
                timestamp = int(time.time())
                # write to file
                write_pmap         (event, S1, S2, Si)
                write_run_and_event(run_number, event, timestamp)

    # Read back
    s1df, s2df, s2sidf = read_pmaps(PMAP_file_name)
    rundf, evtdf = read_run_and_event_from_pmaps_file(PMAP_file_name)
    # get the dicts

    S1L = df_to_pmaps_dict(s1df)
    S2L = df_to_pmaps_dict(s2df)
    S2SiL = df_to_s2si_dict(s2sidf)

    #test
    for event in range(len(XS1L)):
        s1 = XS1L[event]
        if s1: # dictionary not empty
            s1p = S1L[event]
            for peak_number in s1p:
                np.testing.assert_allclose(s1p[peak_number].t,
                                           s1[peak_number][0])
                np.testing.assert_allclose(s1p[peak_number].E,
                                           s1[peak_number][1])
        s2 = XS2L[event]
        if s2:
            s2p = S2L[event]
            for peak_number in s2p:
                np.testing.assert_allclose(s2p[peak_number].t,
                                       s2[peak_number][0])
                np.testing.assert_allclose(s2p[peak_number].E,
                                       s2[peak_number][1])
        s2si = XS2SiL[event]
        if s2si:
            sip = S2SiL[event]
            for peak_number in sip:
                sipm = sip[peak_number]
                sipm2 = s2si[peak_number]
                for nsipm in sipm:
                    np.testing.assert_allclose(sipm[nsipm], sipm2[nsipm])
