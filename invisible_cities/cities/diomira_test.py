"""
code: diomira_test.py
description: test suite for diomira
author: J.J. Gomez-Cadenas
IC core team: Jacek Generowicz, JJGC,
G. Martinez, J.A. Hernando, J.M Benlloch
package: invisible cities. See release notes and licence
"""
import os
import tables as tb
import numpy  as np

from pytest import mark

from .. core import system_of_units as units
from .. core.random_sampling import NoiseSampler as SiPMsNoiseSampler

from .. reco     import tbl_functions as tbl
from .. reco     import wfm_functions as wfm
from .. sierpe   import fee as FEE
from .. sierpe   import blr
from .. database import load_db

from .  diomira  import Diomira


def test_diomira_fee_table(ICDIR):
    """Test that FEE table reads back correctly with expected values."""
    RWF_file = os.path.join(ICDIR,
                            'database/test_data/electrons_40keV_z250_RWF.h5')

    with tb.open_file(RWF_file, 'r') as e40rwf:
        fee = tbl.read_FEE_table(e40rwf.root.MC.FEE)
        feep = fee.fee_param
        eps = 1e-04
        # Ignoring PEP8 to imrpove readability by making symmetry explicit.
        assert len(fee.adc_to_pes)    == e40rwf.root.RD.pmtrwf.shape[1]
        assert len(fee.coeff_blr)     == e40rwf.root.RD.pmtrwf.shape[1]
        assert len(fee.coeff_c)       == e40rwf.root.RD.pmtrwf.shape[1]
        assert len(fee.pmt_noise_rms) == e40rwf.root.RD.pmtrwf.shape[1]
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


def test_diomira_cwf_blr(electron_RWF_file):
    """This is the most rigurous test of the suite. It reads back the
       RWF and BLR waveforms written to disk by DIOMIRA, and computes
       CWF (using deconvoution algorithm), then checks that the CWF match
       the BLR within 1 %.
    """
    with tb.open_file(electron_RWF_file, 'r') as e40rwf:

        pmtrwf = e40rwf.root.RD.pmtrwf
        pmtblr = e40rwf.root.RD.pmtblr
        DataPMT = load_db.DataPMT(0)
        coeff_c = DataPMT.coeff_c.values.astype(np.double)
        coeff_blr = DataPMT.coeff_blr.values.astype(np.double)

        event = 0
        BLR = pmtblr[event]
        CWF = blr.deconv_pmt(pmtrwf[event],
                             coeff_c,
                             coeff_blr,
                             n_baseline=28000,
                             thr_trigger=5)

        diff = wfm.compare_cwf_blr(cwf         = [CWF],
                                   pmtblr      = [BLR],
                                   event_list  = [0],
                                   window_size = 500)
        assert diff[0] < 1


@mark.slow
def test_diomira_sipm(irene_diomira_chain_tmpdir, ICDIR):
    """This test checks that the number of SiPms surviving a hard energy
        cut (50 pes) is always small (<10). The test exercises the full
       construction of the SiPM vectors as well as the noise suppression.
    """
    cal_min = 13
    cal_max = 19
    # the average calibration constant is 16 see diomira_nb in Docs
    sipm_noise_cut = 20 # in pes. Should kill essentially all background

    max_sipm_with_signal = 10
    infile = os.path.join(ICDIR,
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
        for event in range(5):
            # signal in sipm with noise
            sipmrwf = e40rd.root.sipmrd[event] + noise_sampler.Sample()
            # zs waveform
            sipmzs = wfm.noise_suppression(sipmrwf, sipms_thresholds)
            n_sipm = 0
            for j in range(sipmzs.shape[0]):
                if np.sum(sipmzs[j] > 0):
                    n_sipm+=1
            assert n_sipm < max_sipm_with_signal


def test_diomira_identify_bug(ICDIR):
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

    infile = os.path.join(ICDIR,
                          'database/test_data/irene_bug_Kr_ACTIVE_7bar_MCRD.h5')
    with tb.open_file(infile, 'r') as h5in:

        pmtrd  = h5in.root.pmtrd
        pmtwf = pmtrd[0]
        for i in range(pmtrd.shape[1]):
            assert np.sum(pmtwf[i]) == 0


@mark.slow
def test_diomira_copy_mc_and_offset(config_tmpdir):
    PATH_IN = os.path.join(os.environ['ICDIR'],
              'database/test_data/',
              'electrons_40keV_z250_MCRD.h5')
    PATH_OUT = os.path.join(str(config_tmpdir),
              'electrons_40keV_z250_RWF.h5')
    start_evt  = Diomira.event_number_from_input_file_name(str(PATH_IN))
    run_number = 0

    diomira = Diomira(run_number = run_number,
                      files_in   = [PATH_IN],
                      file_out   = PATH_OUT,
                      first_evt  = start_evt)

    nrequired = 2
    nactual = diomira.run(nmax=nrequired)

    if nrequired > 0:
        assert nrequired == nactual

    with tb.open_file(PATH_IN,  mode='r') as h5in, \
         tb.open_file(PATH_OUT, mode='r') as h5out:
            # check event & run number
            assert h5out.root.Run.runInfo[0]['run_number'] == run_number
            assert h5out.root.Run.events [0]['evt_number'] == start_evt

            # check mctracks
            # we have to convert manually into a tuple because MCTracks[0]
            # returns an object of type numpy.void where we cannot index
            # using ranges like mctracks_in[1:]
            mctracks_in  = tuple(h5in .root.MC.MCTracks[0])
            mctracks_out = tuple(h5out.root.MC.MCTracks[0])
            #evt number is not equal if we redefine first event number
            assert mctracks_out[0] == start_evt
            for e in zip(mctracks_in[1:], mctracks_out[1:]):
                np.testing.assert_array_equal(e[0],e[1])

@mark.parametrize('filename, first_evt',
             (('dst_NEXT_v0_08_09_Co56_INTERNALPORTANODE_74_0_7bar_MCRD_10000.root.h5',
               740000),
              ('NEXT_v0_08_09_Co56_2_0_7bar_MCRD_1000.root.h5',
               2000),
              ('electrons_40keV_z250_MCRD.h5',
               0)))
def test_event_number_from_input_file_name(filename, first_evt):
    assert Diomira.event_number_from_input_file_name(filename) == first_evt

