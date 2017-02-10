"""
code: isidora_test.py
description: test suite for isidora
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

from pytest import mark, raises

from   invisible_cities.core.configure import configure
from   invisible_cities.core.exceptions import NoInputFiles
import invisible_cities.core.tbl_functions as tbl
import invisible_cities.core.system_of_units as units
from   invisible_cities.sierpe import fee as FEE
import invisible_cities.sierpe.blr as blr
from   invisible_cities.database import load_db
from   invisible_cities.cities.isidora import Isidora
from   invisible_cities.core.random_sampling \
     import NoiseSampler as SiPMsNoiseSampler

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

# run
NEVENTS {NEVENTS}
RUN_ALL {RUN_ALL}
"""


@mark.slow
def test_isidora_run(config_tmpdir):
    """Test that Isidora runs on default config parameters.
       Check that CWF waveforms math BLR within 1%. """

    # Create config file for Isidora.
    config_file_spec = dict(PATH_IN  = '$ICDIR/database/test_data/',
                            FILE_IN  = 'electrons_40keV_z250_RWF.h5',
                            PATH_OUT = str(config_tmpdir),
                            FILE_OUT = 'electrons_40keV_z250_CWF.h5',
                            COMPRESSION = 'ZLIB4',
                            RUN_NUMBER  =     0,
                            NPRINT      =     1,
                            NBASELINE   = 28000,
                            THR_TRIGGER =     5,
                            NMAU        =   100,
                            THR_MAU     =     3,
                            NEVENTS     =     5,
                            RUN_ALL     = False)

    config_file_contents = config_file_format.format(**config_file_spec)
    conf_file_name = str(config_tmpdir.join('test.conf'))
    with open(conf_file_name, 'w') as conf_file:
        conf_file.write(config_file_contents)

    # Read back config file and run Isidora.
    CFP = configure(['ISIDORA', '-c', conf_file_name])
    RWF_file = CFP['FILE_IN']
    CWF_file = CFP['FILE_OUT']

    fpp = Isidora(run_number=CFP['RUN_NUMBER'])
    files_in = glob(RWF_file)
    files_in.sort()
    fpp.set_input_files(files_in)
    fpp.set_cwf_store(CWF_file, compression = CFP['COMPRESSION'])
    fpp.set_print(nprint = CFP['NPRINT'])
    fpp.set_BLR(n_baseline  = CFP['NBASELINE'],
                thr_trigger = CFP['THR_TRIGGER'] * units.adc)
    fpp.set_MAU(  n_MAU = CFP['NMAU'],
                thr_MAU = CFP['THR_MAU'] * units.adc)

    nevts = CFP['NEVENTS'] if not CFP['RUN_ALL'] else -1
    nevt = fpp.run(nmax=nevts)
    assert nevt == nevts

    # Check CWF and BLR waveforms match within 1% in the region of interest.
    eps = 1
    peak_start = 4950
    peak_end   = 5150
    with tb.open_file(RWF_file, 'r') as h5in, tb.open_file(CWF_file, 'r') as h5out:
        pmtblr = h5in. root.RD. pmtblr
        pmtcwf = h5out.root.BLR.pmtcwf
        pmtblr_roi = pmtblr[ :nevt, : , peak_start:peak_end]
        pmtcwf_roi = pmtcwf[ :nevt, : , peak_start:peak_end]
        pmtblr_sum = np.sum(pmtblr_roi, axis=2)
        pmtcwf_sum = np.sum(pmtcwf_roi, axis=2)
        np.testing.assert_allclose(pmtcwf_sum, pmtblr_sum, rtol=0.01, atol=0)


def test_isidora_no_input():
    """Test that Isidora throws NoInputFiles exceptions if no input files
       are defined. """

    with raises(NoInputFiles):
        fpp = Isidora(run_number = 0)
        fpp.run(nmax=1)
