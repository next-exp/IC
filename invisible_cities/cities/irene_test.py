"""
code: irene_test.py
description: test suite for irene
author: J.J. Gomez-Cadenas
IC core team: Jacek Generowicz, JJGC,
G. Martinez, J.A. Hernando, J.M Benlloch
package: invisible cities. See release notes and licence
last changed: 01-12-2017
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
import invisible_cities.sierpe.blr as blr
from   invisible_cities.database import load_db
from   invisible_cities.cities.diomira import Diomira
from   invisible_cities.cities.irene import Irene
from   invisible_cities.core.random_sampling \
     import NoiseSampler as SiPMsNoiseSampler


def test_diomira_and_irene_run():
    """Test that Diomira & Irene runs on default config parameters."""
    conf_file = os.environ['ICDIR'] + '/config/diomira.conf'
    CFP = configure(['DIOMIRA','-c', conf_file])
    fpp = Diomira()
    files_in = glob(CFP['FILE_IN'])
    files_in.sort()
    fpp.set_input_files(files_in)
    fpp.set_output_file(CFP['FILE_OUT'],
                        compression = CFP['COMPRESSION'])
    fpp.set_print(nprint = CFP['NPRINT'])
    fpp.set_sipm_noise_cut(noise_cut = CFP["NOISE_CUT"])

    nevts = CFP['NEVENTS'] if not CFP['RUN_ALL'] else -1
    nevt = fpp.run(nmax=nevts)
    assert nevt == nevts

    # Irene
    conf_file = os.environ['ICDIR'] + '/config/irene.conf'
    CFP = configure(['IRENE', '-c', conf_file])

    fpp = Irene(run_number=CFP['RUN_NUMBER'])

    files_in = glob(CFP['FILE_IN'])
    files_in.sort()
    fpp.set_input_files(files_in)
    fpp.set_pmap_store(CFP['FILE_OUT'],
                       compression = CFP['COMPRESSION'])
    fpp.set_print(nprint=CFP['NPRINT'])

    fpp.set_BLR(n_baseline  = CFP['NBASELINE'],
                thr_trigger = CFP['THR_TRIGGER'] * units.adc)

    fpp.set_MAU(  n_MAU = CFP['NMAU'],
                thr_MAU = CFP['THR_MAU'] * units.adc)

    fpp.set_CSUM(thr_csum=CFP['THR_CSUM'] * units.pes)

    fpp.set_S1(tmin   = CFP['S1_TMIN'] * units.mus,
               tmax   = CFP['S1_TMAX'] * units.mus,
               stride = CFP['S1_STRIDE'],
               lmin   = CFP['S1_LMIN'],
               lmax   = CFP['S1_LMAX'])

    fpp.set_S2(tmin   = CFP['S2_TMIN'] * units.mus,
               tmax   = CFP['S2_TMAX'] * units.mus,
               stride = CFP['S2_STRIDE'],
               lmin   = CFP['S2_LMIN'],
               lmax   = CFP['S2_LMAX'])

    fpp.set_SiPM(thr_zs=CFP['THR_ZS'] * units.pes,
                 thr_sipm_s2=CFP['THR_SIPM_S2'] * units.pes)

    nevts = CFP['NEVENTS'] if not CFP['RUN_ALL'] else -1
    nevt = fpp.run(nmax=nevts, store_pmaps=True)
    assert nevt == nevts

    # This leads to the dark side !
    os.system('rm $ICDIR/database/test_data/electrons_40keV_z250_RWF.h5')
    os.system('rm $ICDIR/database/test_data/electrons_40keV_z250_PMP.h5')
