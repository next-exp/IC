"""
code: cecilia_test.py
description: test suite for trigger city
author: P. Ferrario
IC core team: Jacek Generowicz, JJGC,
G. Martinez, J.A. Hernando, J.M Benlloch
package: invisible cities. See release notes and licence
"""
import os
import tables as tb
import numpy as np

from pytest import mark, fixture

from invisible_cities.cities.cecilia import Cecilia, CECILIA

@fixture(scope='module')
def conf_file_name(config_tmpdir):
    # Specifies a name for a MC configuration file. Also, default number
    # of events set to 1.

    conf_file_name = str(config_tmpdir.join('cecilia_test.conf'))
    Cecilia.write_config_file(conf_file_name,
                              PATH_OUT = str(config_tmpdir),
                              COMPRESSION = "ZLIB4",
                              NEVENTS = 130,
                              FIRST_EVT = 0,
                              TR_CHANNELS = "0 1 2 3 4 5 6 7 8 9 10 11",
                              MIN_NUMB_CHANNELS = 5,
                              MIN_HEIGHT = 15,
                              MAX_HEIGHT = 1000,
                              MIN_CHARGE = 3000,
                              MAX_CHARGE = 20000,
                              MIN_WIDTH = 4000,
                              MAX_WIDTH = 12000,
                              DATA_MC_RATIO = 0.8)
    return conf_file_name

def test_command_line_trigger_krypton(conf_file_name, config_tmpdir, ICDIR):
    # NB: avoid taking defaults for PATH_IN and PATH_OUT
    # since they are in general test-specific
    # NB: avoid taking defaults for run number (test-specific)

    PATH_IN = os.path.join(ICDIR,
                           'database/test_data/',
                           'kr_cwf.root.h5')
    PATH_OUT = os.path.join(str(config_tmpdir),
                            'kr_trigwf.h5')

    nevts, n_evts_run = CECILIA(['CECILIA',
                                 '-c', conf_file_name,
                                 '-i', PATH_IN,
                                 '-o', PATH_OUT,
                                 '-n', '20',
                                 '-r', '0'])
    if nevts != -1:
        assert nevts == n_evts_run
        
