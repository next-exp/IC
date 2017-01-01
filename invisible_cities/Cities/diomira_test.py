from __future__ import print_function
from __future__ import absolute_import

from glob import glob
import tables as tb
import numpy as np

from invisible_cities.Core.Configure import configure
import invisible_cities.Core.tblFunctions as tbl
import invisible_cities.Core.system_of_units as units
from invisible_cities.Sierpe import FEE
import invisible_cities.ICython.Sierpe.BLR as blr
from invisible_cities.Database import loadDB
from invisible_cities.Cities.diomira_ms import Diomira
import os

def test_diomira_run():
    """ Tests that DIOMIRA runs on default config parameters """
    ffile = os.environ['ICDIR'] + '/invisible_cities/tests/electrons_40keV_z250_RWF.h5'
    try:
        os.system("rm -f {}".format(ffile))
    except(IOError):
        pass

    ffile = os.environ['ICDIR'] + '/invisible_cities/Config/diomira_ms.conf'
    CFP = configure(['DIOMIRA','-c',ffile])
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

def test_diomira_fee_table():
    """ tests that FEE table reads back correctly with expected values"""
    path = os.environ['ICDIR'] + '/invisible_cities/tests/'

    ffile ='electrons_40keV_z250_RWF.h5'
    e40rwf= tb.open_file(path+ffile,'r+')
    fee = tbl.read_FEE_table(e40rwf.root.MC.FEE)
    feep = fee['fee_param']
    eps = 1e-04
    assert len(fee['adc_to_pes']) == e40rwf.root.RD.pmtrwf.shape[1]
    assert len(fee['coeff_blr']) == e40rwf.root.RD.pmtrwf.shape[1]
    assert len(fee['coeff_c']) == e40rwf.root.RD.pmtrwf.shape[1]
    assert len(fee['pmt_noise_rms']) == e40rwf.root.RD.pmtrwf.shape[1]
    assert abs(feep.FEE_GAIN - FEE.FEE_GAIN) < eps
    assert feep.NBITS == FEE.NBITS
    assert abs(feep.LSB - FEE.LSB) < eps
    assert abs(feep.NOISE_I - FEE.NOISE_I) < eps
    assert abs(feep.NOISE_DAQ - FEE.NOISE_DAQ) < eps
    assert abs(feep.C2/units.nF - FEE.C2/units.nF) < eps
    assert abs(feep.C1/units.nF - FEE.C1/units.nF) < eps
    assert  abs(feep.R1/units.ohm - FEE.R1/units.ohm) < eps
    assert  abs(feep.ZIN/units.ohm - FEE.Zin/units.ohm) < eps
    assert  abs(feep.t_sample - FEE.t_sample) < eps
    assert  abs(feep.f_sample - FEE.f_sample) < eps
    assert  abs(feep.f_mc - FEE.f_mc) < eps
    assert  abs(feep.f_LPF1 - FEE.f_LPF1) < eps
    assert  abs(feep.f_LPF2 - FEE.f_LPF2) < eps
    assert  abs(feep.OFFSET - FEE.OFFSET) < eps
    assert  abs(feep.CEILING - FEE.CEILING) < eps

def test_diomira_cwf_blr():
    """This is the most rigurous test of the suite. It reads back the
       RWF and BLR waveforms written to disk by DIOMIRA, and computes
       CWF (using deconvoution algorithm), then checks that the CWF match
       the BLR within 1 %.
    """
    eps = 1.
    path = os.environ['ICDIR'] + '/invisible_cities/tests/'
    ffile ='electrons_40keV_z250_RWF.h5'
    e40rwf= tb.open_file(path+ffile,'r+')
    pmtrwf = e40rwf.root.RD.pmtrwf
    pmtblr = e40rwf.root.RD.pmtblr
    DataPMT = loadDB.DataPMT(0)
    coeff_c = DataPMT.coeff_c.values.astype(np.double)
    coeff_blr = DataPMT.coeff_blr.values.astype(np.double)

    for event in range(10):
        BLR = pmtblr[event]
        CWF = blr.deconv_pmt(pmtrwf[event], coeff_c, coeff_blr,
                     n_baseline=28000, thr_trigger=5)

        for i in range(len(CWF)):
            diff = abs(np.sum(BLR[i][5000:5100]) - np.sum(CWF[i][5000:5100]))
            diff = 100. * diff/np.sum(BLR[i])
            assert diff < eps
