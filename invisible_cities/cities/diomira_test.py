"""
code: diomira_test.py
description: test suite for diomira
author: J.J. Gomez-Cadenas
IC core team: Jacek Generowicz, JJGC, G. Martinez, J.A. Hernando, J.M Benlloch
package: invisible cities. See release notes and licence
last changed: 09-11-2017
"""
from __future__ import print_function
from __future__ import absolute_import

import os
from glob import glob
import tables as tb
import numpy as np

import pytest

from   invisible_cities.core.configure import configure
import invisible_cities.core.tbl_functions as tbl
import invisible_cities.core.wfm_functions as wfm
import invisible_cities.core.system_of_units as units
from   invisible_cities.sierpe import fee as FEE
import invisible_cities.sierpe.BLR as blr
from   invisible_cities.database import load_db
from   invisible_cities.cities.diomira import Diomira
from   invisible_cities.core.random_sampling \
     import NoiseSampler as SiPMsNoiseSampler

@pytest.fixture(scope='session')
def electrons_40keV_z250_RWF_h5(tmpdir_factory):
    return (str(tmpdir_factory
                .mktemp('diomira_tests')
                .join('electrons_40keV_z250_RWF.h5')))

def test_diomira_run(electrons_40keV_z250_RWF_h5):
    """Test that DIOMIRA runs on default config parameters."""

    conf_file = os.environ['ICDIR'] + '/config/diomira.conf'
    CFP = configure(['DIOMIRA','-c', conf_file, '-o', electrons_40keV_z250_RWF_h5])
    fpp = Diomira()
    files_in = glob(CFP['FILE_IN'])
    files_in.sort()
    fpp.set_input_files(files_in)
    fpp.set_output_file(CFP['FILE_OUT'],
                        compression=CFP['COMPRESSION'])
    fpp.set_print(nprint=CFP['NPRINT'])

    fpp.set_sipm_noise_cut(noise_cut=CFP["NOISE_CUT"])

    nevts = CFP['NEVENTS'] if not CFP['RUN_ALL'] else -1
    nevt = fpp.run(nmax=nevts)

    assert nevt == nevts

def test_diomira_fee_table(electrons_40keV_z250_RWF_h5):
    """Test that FEE table reads back correctly with expected values."""

    with tb.open_file(electrons_40keV_z250_RWF_h5, 'r+') as e40rwf:
        fee = tbl.read_FEE_table(e40rwf.root.MC.FEE)
        feep = fee['fee_param']
        eps = 1e-04
        # Ignoring PEP8 to imrpove readability by making symmetry explicit.
        assert len(fee['adc_to_pes'])    == e40rwf.root.RD.pmtrwf.shape[1]
        assert len(fee['coeff_blr'])     == e40rwf.root.RD.pmtrwf.shape[1]
        assert len(fee['coeff_c'])       == e40rwf.root.RD.pmtrwf.shape[1]
        assert len(fee['pmt_noise_rms']) == e40rwf.root.RD.pmtrwf.shape[1]
        assert feep.NBITS == FEE.NBITS
        assert abs(feep.FEE_GAIN - FEE.FEE_GAIN)           < eps
        assert abs(feep.LSB - FEE.LSB)                     < eps
        assert abs(feep.NOISE_I - FEE.NOISE_I)             < eps
        assert abs(feep.NOISE_DAQ - FEE.NOISE_DAQ)         < eps
        assert abs(feep.C2/units.nF - FEE.C2/units.nF)     < eps
        assert abs(feep.C1/units.nF - FEE.C1/units.nF)     < eps
        assert abs(feep.R1/units.ohm - FEE.R1/units.ohm)   < eps
        assert abs(feep.ZIN/units.ohm - FEE.Zin/units.ohm) < eps
        assert abs(feep.t_sample - FEE.t_sample)           < eps
        assert abs(feep.f_sample - FEE.f_sample)           < eps
        assert abs(feep.f_mc - FEE.f_mc)                   < eps
        assert abs(feep.f_LPF1 - FEE.f_LPF1)               < eps
        assert abs(feep.f_LPF2 - FEE.f_LPF2)               < eps
        assert abs(feep.OFFSET - FEE.OFFSET)               < eps
        assert abs(feep.CEILING - FEE.CEILING)             < eps


def test_diomira_cwf_blr(electrons_40keV_z250_RWF_h5):
    """This is the most rigurous test of the suite. It reads back the
       RWF and BLR waveforms written to disk by DIOMIRA, and computes
       CWF (using deconvoution algorithm), then checks that the CWF match
       the BLR within 1 %.
    """
    eps = 1
    with tb.open_file(electrons_40keV_z250_RWF_h5, 'r+') as e40rwf:

        pmtrwf = e40rwf.root.RD.pmtrwf
        pmtblr = e40rwf.root.RD.pmtblr
        DataPMT = load_db.DataPMT(0)
        coeff_c = DataPMT.coeff_c.values.astype(np.double)
        coeff_blr = DataPMT.coeff_blr.values.astype(np.double)

        for event in range(10):
            BLR = pmtblr[event]
            CWF = blr.deconv_pmt(pmtrwf[event],
                                 coeff_c,
                                 coeff_blr,
                                 n_baseline=28000,
                                 thr_trigger=5)

            for i in range(len(CWF)):
                diff = abs(np.sum(BLR[i][5000:5100]) - np.sum(CWF[i][5000:5100]))
                diff = 100. * diff/np.sum(BLR[i])
                assert diff < eps


def test_diomira_sipm(electrons_40keV_z250_RWF_h5):
    """This test checks that the number of SiPms surviving a hard energy
        cut (50 pes) is always small (<10). The test exercises the full
       construction of the SiPM vectors as well as the noise suppression.
    """
    cal_min = 13
    cal_max = 19
    # the average calibration constant is 16 see diomira_nb in Docs
    sipm_noise_cut = 20 # in pes. Should kill essentially all background

    max_sipm_with_signal = 10
    infile = os.environ['ICDIR'] + '/database/test_data/electrons_40keV_z250_MCRD.h5'
    with tb.open_file(infile, 'r+') as e40rd:

        NEVENTS_DST, NSIPM, SIPMWL = e40rd.root.sipmrd.shape

        assert NSIPM == 1792
        assert SIPMWL == 800

        DataSiPM = load_db.DataSiPM(0)
        sipm_adc_to_pes = DataSiPM.adc_to_pes.values.astype(np.double)

        # check that the mean of (non zero) SiPMs is within reasonable values
        # NB: this tests could fail if the average calibration constants in the
        # data change dramatically, but in this case is interesting to check and
        # redefine boundaries
        assert np.mean(sipm_adc_to_pes[sipm_adc_to_pes>0]) > cal_min
        assert np.mean(sipm_adc_to_pes[sipm_adc_to_pes>0]) < cal_max

        sipms_thresholds = sipm_noise_cut *  sipm_adc_to_pes

        noise_sampler = SiPMsNoiseSampler(SIPMWL, True)
        for event in range(10):
            # signal in sipm with noise
            sipmrwf = e40rd.root.sipmrd[event] + noise_sampler.Sample()
            # zs waveform
            sipmzs = wfm.noise_suppression(sipmrwf, sipms_thresholds)
            n_sipm = 0
            for j in range(sipmzs.shape[0]):
                if np.sum(sipmzs[j] >0):
                    n_sipm+=1
            assert n_sipm < max_sipm_with_signal
