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
from os import path
from glob import glob
import tables as tb
import numpy as np

from pytest import mark

from   invisible_cities.core.configure import configure
import invisible_cities.core.tbl_functions as tbl
import invisible_cities.core.system_of_units as units
from   invisible_cities.sierpe import fee as FEE
import invisible_cities.sierpe.blr as blr
from   invisible_cities.database import load_db
from   invisible_cities.cities.diomira import Diomira
from   invisible_cities.cities.irene import Irene, S12Params as S12P, IRENE
from   invisible_cities.core.random_sampling \
     import NoiseSampler as SiPMsNoiseSampler


@mark.slow
def test_diomira_and_irene_run(irene_diomira_chain_tmpdir):
    """Test that Diomira & Irene runs on default config parameters."""

    MCRD_file = path.join(os.environ['ICDIR'],
                          'database/test_data/electrons_40keV_z250_MCRD.h5')

    RWF_file = str(irene_diomira_chain_tmpdir.join(
                   'electrons_40keV_z250_RWF.h5'))
    conf_file = path.join(os.environ['ICDIR'], 'config/diomira.conf')
    CFP = configure(['DIOMIRA',
                     '-c', conf_file,
                     '-i', MCRD_file,
                     '-o', RWF_file,
                     '-n', '5'])
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
    conf_file = path.join(os.environ['ICDIR'], 'config/irene.conf')
    PMP_file = str(irene_diomira_chain_tmpdir.join(
                   'electrons_40keV_z250_PMP.h5'))
    CFP = configure(['IRENE',
                     '-c', conf_file,
                     '-i', RWF_file,
                     '-o', PMP_file,
                     '-n', '5'])

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

    fpp.set_s12(s1 = S12P(tmin   = CFP['S1_TMIN'] * units.mus,
                          tmax   = CFP['S1_TMAX'] * units.mus,
                          stride = CFP['S1_STRIDE'],
                          lmin   = CFP['S1_LMIN'],
                          lmax   = CFP['S1_LMAX']),
                s2 = S12P(tmin   = CFP['S2_TMIN'] * units.mus,
                          tmax   = CFP['S2_TMAX'] * units.mus,
                          stride = CFP['S2_STRIDE'],
                          lmin   = CFP['S2_LMIN'],
                          lmax   = CFP['S2_LMAX']))

    fpp.set_sipm(thr_zs=CFP['THR_ZS'] * units.pes,
                 thr_sipm_s2=CFP['THR_SIPM_S2'] * units.pes)

    nevts = CFP['NEVENTS'] if not CFP['RUN_ALL'] else -1
    nevt = fpp.run(nmax=nevts)
    assert nevt == nevts

def test_empty_events(irene_diomira_chain_tmpdir):
    """Test Irene on a file containing an empty event."""

    RWF_file = path.join(os.environ['ICDIR'],
                         'database/test_data/irene_bug_Kr_ACTIVE_7bar_RWF.h5')
    conf_file = path.join(os.environ['ICDIR'], 'config/irene.conf')

    PMP_file = str(irene_diomira_chain_tmpdir.join(
                  'electrons_40keV_z250_PMP.h5'))

    CFP = configure(['IRENE',
                             '-c', conf_file,
                             '-i', RWF_file,
                             '-o', PMP_file,
                             '-n', '5'])

    fpp = Irene(run_number = CFP['RUN_NUMBER'])

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

    fpp.set_s12(s1 = S12P(tmin   = CFP['S1_TMIN'] * units.mus,
                          tmax   = CFP['S1_TMAX'] * units.mus,
                          stride = CFP['S1_STRIDE'],
                          lmin   = CFP['S1_LMIN'],
                          lmax   = CFP['S1_LMAX']),
                s2 = S12P(tmin   = CFP['S2_TMIN'] * units.mus,
                          tmax   = CFP['S2_TMAX'] * units.mus,
                          stride = CFP['S2_STRIDE'],
                          lmin   = CFP['S2_LMIN'],
                          lmax   = CFP['S2_LMAX']))

    fpp.set_sipm(thr_zs      = CFP['THR_ZS']      * units.pes,
                 thr_sipm_s2 = CFP['THR_SIPM_S2'] * units.pes)

    nevts = CFP['NEVENTS'] if not CFP['RUN_ALL'] else -1
    nevt = fpp.run(nmax=nevts)
    assert fpp.empty_events == 1 # found one empty event
    assert nevt == 0 # event did not process




# TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO
#
# The test config file creation is duplicated here and in isidora_test
# (and in maurilia). This duplication must be eliminated.

config_file_format = """
# set_input_files
PATH_IN {PATH_IN}
FILE_IN {FILE_IN}

# set_cwf_store
PATH_OUT {PATH_OUT}
FILE_OUT {FILE_OUT}
COMPRESSION {COMPRESSION}

# irene
RUN_NUMBER {RUN_NUMBER}

# set_print
NPRINT {NPRINT}

# set_blr
NBASELINE {NBASELINE}
THR_TRIGGER {THR_TRIGGER}

# set_mau
NMAU {NMAU}
THR_MAU {THR_MAU}

# set_csum
THR_CSUM 1

# set_s1
S1_TMIN {S1_TMIN}
S1_TMAX {S1_TMAX}
S1_STRIDE {S1_STRIDE}
S1_LMIN {S1_LMIN}
S1_LMAX {S1_LMAX}

# set_s2
S2_TMIN {S2_TMIN}
S2_TMAX {S2_TMAX}
S2_STRIDE {S2_STRIDE}
S2_LMIN {S2_LMIN}
S2_LMAX {S2_LMAX}

# set_sipm
THR_ZS {THR_ZS}
THR_SIPM_S2 {THR_SIPM_S2}

# run
PRINT_EMPTY_EVENTS {PRINT_EMPTY_EVENTS}
NEVENTS {NEVENTS}
RUN_ALL {RUN_ALL}
"""

def config_file_spec_with_tmpdir(tmpdir):
    return dict(PATH_IN  = '$ICDIR/database/test_data/',
                FILE_IN  = 'electrons_40keV_z250_RWF.h5',
                PATH_OUT = str(tmpdir),
                FILE_OUT = 'electrons_40keV_z250_PMP.h5',
                COMPRESSION        = 'ZLIB4',
                RUN_NUMBER         =      0,
                NPRINT             =      1,
                NBASELINE          =  28000,
                THR_TRIGGER        =      5,
                NMAU               =    100,
                THR_MAU            =      3,
                THR_CSUM           =      0.5,
                S1_TMIN            =     10,
                S1_TMAX            =    590,
                S1_STRIDE          =      4,
                S1_LMIN            =     10,
                S1_LMAX            =     16,
                S2_TMIN            =    110,
                S2_TMAX            =   1190,
                S2_STRIDE          =     40,
                S2_LMIN            =    100,
                S2_LMAX            = 100000,
                THR_ZS             =     10,
                THR_SIPM_S2        =     30,
                PRINT_EMPTY_EVENTS = True,
                NEVENTS            =      5,                
                RUN_ALL            = False)

    
# TODO refactor to factor out config file creation: most of this test
# is noise; duplication of something that also happens in the above
# test
def test_command_line_irene(config_tmpdir):

    config_file_spec = config_file_spec_with_tmpdir(config_tmpdir)

    config_file_contents = config_file_format.format(**config_file_spec)
    conf_file_name = str(config_tmpdir.join('test-3.conf'))
    with open(conf_file_name, 'w') as conf_file:
        conf_file.write(config_file_contents)

    IRENE(['IRENE', '-c', conf_file_name])

