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

config_file_format = """

# set_input_files
PATH_IN {PATH_IN}
FILE_IN {FILE_IN}

# set_cwf_store
PATH_OUT {PATH_OUT}
FILE_OUT {FILE_OUT}
COMPRESSION {COMPRESSION}
"""

config_file_spec = dict(PATH_IN  = '$ICDIR/database/test_data/',
                        FILE_IN  = 'NEW_se_mc_reco_pmaps_1evt.h5',
                        PATH_OUT = '$ICDIR/database/test_data/',
                        FILE_OUT = 'NEW_se_mc_1evt.h5',
                        COMPRESSION = 'ZLIB4')

config_file_contents = config_file_format.format(**config_file_spec)

@mark.serial
@mark.slow
def test_maurilia_run(config_tmpdir):
    """Test that MAURILLA runs on default config parameters."""

    conf_file_name = str(config_tmpdir.join('maurilia_test.conf'))
    with open(conf_file_name, 'w') as conf_file:
        conf_file.write(config_file_contents)

    conf_file = path.join(os.environ['ICDIR'], 'config/maurilia.conf')
    CFP = configure(['MAURILLA','-c', conf_file])
    fpp = Maurilia()
    files_in = glob(CFP['FILE_IN'])
    files_in.sort()
    fpp.set_input_files(files_in)
    fpp.set_output_file(CFP['FILE_OUT'],
                        compression=CFP['COMPRESSION'])

    fpp.run()

    RWF_file = CFP['FILE_OUT']
    with tb.open_file(RWF_file, 'r') as fmc:
        mctbl = fmc.root.MC.MCTracks
        assert len(mctbl)    == 283
