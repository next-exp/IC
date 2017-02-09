"""
code: maurilia_test.py
description: test suite for maurilia
author: Josh Renner
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
from   invisible_cities.cities.maurilia import Maurilia
from   invisible_cities.core.random_sampling \
     import NoiseSampler as SiPMsNoiseSampler

@mark.slow
def test_maurilia_run(config_tmpdir):
    """Test that MAURILIA runs on default config parameters."""

    # set up the test configuration file
    config_file_spec = config_file_spec_with_tmpdir(config_tmpdir)
    config_file_contents = config_file_format.format(**config_file_spec)
    conf_file_name = str(config_tmpdir.join('test-maurilia.conf'))
    with open(conf_file_name, 'w') as conf_file:
        conf_file.write(config_file_contents)

    # configure the Maurilia class
    CFP = configure(['MAURILLA', '-c', conf_file_name])
    fpp = Maurilia()
    files_in = glob(CFP['FILE_IN'])
    files_in.sort()
    fpp.set_input_files(files_in)
    fpp.set_output_file(CFP['FILE_OUT'])
    fpp.set_compression(CFP['COMPRESSION'])

    # run Maurilia
    fpp.run()

    # perform the test
    with tb.open_file(CFP['FILE_OUT'], 'r') as fmc:
        mctbl = fmc.root.MC.MCTracks
        assert len(mctbl)    == 283


config_file_format = """
# set_input_files
PATH_IN {PATH_IN}
FILE_IN {FILE_IN}

# set_cwf_store
PATH_OUT {PATH_OUT}
FILE_OUT {FILE_OUT}
COMPRESSION {COMPRESSION}
"""

def config_file_spec_with_tmpdir(tmpdir):
    return dict(PATH_IN  = '$ICDIR/database/test_data/',
                FILE_IN  = 'NEW_se_mc_reco_pmaps_1evt.h5',
                PATH_OUT = str(tmpdir),
                FILE_OUT = 'NEW_se_mc_1evt.h5',
                COMPRESSION = 'ZLIB4')
