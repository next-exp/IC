"""
code: diomira_test.py
description: test suite for diomira
author: J.J. Gomez-Cadenas
IC core team: Jacek Generowicz, JJGC,
G. Martinez, J.A. Hernando, J.M Benlloch
package: invisible cities. See release notes and licence
"""
from __future__ import print_function
from __future__ import absolute_import

import os
from os import path
from glob import glob
import tables as tb
import numpy as np

from pytest import mark

from   invisible_cities.core.configure import configure
import invisible_cities.core.tbl_functions as tbl
import invisible_cities.core.wfm_functions as wfm
import invisible_cities.core.system_of_units as units
from   invisible_cities.sierpe import fee as FEE
import invisible_cities.sierpe.blr as blr
from   invisible_cities.database import load_db
from   invisible_cities.cities.diomira import Diomira, DIOMIRA
from   invisible_cities.core.random_sampling \
     import NoiseSampler as SiPMsNoiseSampler


@mark.serial
@mark.slow
def test_diomira_run(irene_diomira_chain_tmpdir):
    """Test that DIOMIRA runs on default config parameters."""
    RWF_file = str(irene_diomira_chain_tmpdir.join(
                   'electrons_40keV_z250_RWF.h5'))
    conf_file = path.join(os.environ['ICDIR'], 'config/diomira.conf')
    CFP = configure(['DIOMIRA','-c', conf_file, '-o', RWF_file])
    fpp = Diomira()
    files_in = glob(CFP['FILE_IN'])
    files_in.sort()
    fpp.set_input_files(files_in)
    fpp.set_output_file(CFP['FILE_OUT'], compression=CFP['COMPRESSION'])
    fpp.set_print(nprint=CFP['NPRINT'])

    fpp.set_sipm_noise_cut(noise_cut=CFP["NOISE_CUT"])

    nevts = CFP['NEVENTS'] if not CFP['RUN_ALL'] else -1
    nevt = fpp.run(nmax=nevts)

    assert nevt == nevts

@mark.serial
@mark.slow # not slow in itself, but in series with test_diomira_run
def test_diomira_fee_table(irene_diomira_chain_tmpdir):
    """Test that FEE table reads back correctly with expected values."""
    RWF_file = str(irene_diomira_chain_tmpdir.join(
                   'electrons_40keV_z250_RWF.h5'))
    with tb.open_file(RWF_file, 'r') as e40rwf:
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


@mark.serial
@mark.slow # not slow in itself, but in series with test_diomira_run
def test_diomira_cwf_blr(irene_diomira_chain_tmpdir):
    """This is the most rigurous test of the suite. It reads back the
       RWF and BLR waveforms written to disk by DIOMIRA, and computes
       CWF (using deconvoution algorithm), then checks that the CWF match
       the BLR within 1 %.
    """
    eps = 1
    RWF_file = str(irene_diomira_chain_tmpdir.join(
                   'electrons_40keV_z250_RWF.h5'))
    with tb.open_file(RWF_file, 'r') as e40rwf:

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
                diff = 100. * diff / np.sum(BLR[i])
                assert diff < eps


@mark.slow
def test_diomira_sipm(irene_diomira_chain_tmpdir):
    """This test checks that the number of SiPms surviving a hard energy
        cut (50 pes) is always small (<10). The test exercises the full
       construction of the SiPM vectors as well as the noise suppression.
    """
    cal_min = 13
    cal_max = 19
    # the average calibration constant is 16 see diomira_nb in Docs
    sipm_noise_cut = 20 # in pes. Should kill essentially all background

    max_sipm_with_signal = 10
    infile = os.path.join(os.environ['ICDIR'],
                          'database/test_data/electrons_40keV_z250_MCRD.h5')
    with tb.open_file(infile, 'r') as e40rd:

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

def test_diomira_identify_bug():
    """Read a one-event file in which the energy of PMTs is equal to zero and
    asset it must be son. This test would fail for a normal file where there
    is always some energy in the PMTs. It's purpose is to provide an automaic
    documentation of a problem (not really a bug but a feature of the
    simulation) that can cause a crash in some events. NEXUS simulation
    can produce eventually events where none of the PMTs of the EP record
    energy. This test simply ensures that the event read in the example file
    in indeed of this type.

    The same event is later processed with Irene (where a protection
    that skips empty events has been added) to ensure that no crash occur."""

    infile = path.join(os.environ['ICDIR'],
                       'database/test_data/irene_bug_Kr_ACTIVE_7bar_MCRD.h5')
    with tb.open_file(infile, 'r') as h5in:

        pmtrd  = h5in.root.pmtrd
        pmtwf = pmtrd[0]
        for i in range(pmtrd.shape[1]):
            assert np.sum(pmtwf[i]) == 0



config_file_format = """
# set_input_files
PATH_IN {PATH_IN}
FILE_IN {FILE_IN}

# set_cwf_store
PATH_OUT {PATH_OUT}
FILE_OUT {FILE_OUT}
COMPRESSION {COMPRESSION}

# set_print
NPRINT {NPRINT}

# run
NEVENTS {NEVENTS}
RUN_ALL {RUN_ALL}

NOISE_CUT {NOISE_CUT}
"""


def config_file_spec_with_tmpdir(tmpdir):
    return dict(PATH_IN  = '$ICDIR/database/test_data/',
                FILE_IN  = 'electrons_40keV_z250_MCRD.h5',
                PATH_OUT = str(tmpdir),
                FILE_OUT = 'electrons_40keV_z250_CWF.h5',
                COMPRESSION = 'ZLIB4',
                NPRINT      =     1,
                NEVENTS     =     5,
                NOISE_CUT   =     3,
                RUN_ALL     = False)



# TODO refactor to factor out config file creation: most of this test
# is noise; duplication of something that also happens in the above
# test
def test_command_line_diomira(config_tmpdir):

    config_file_spec = config_file_spec_with_tmpdir(config_tmpdir)

    config_file_contents = config_file_format.format(**config_file_spec)
    conf_file_name = str(config_tmpdir.join('test-4.conf'))
    with open(conf_file_name, 'w') as conf_file:
        conf_file.write(config_file_contents)

    DIOMIRA(['DIOMIRA', '-c', conf_file_name])
