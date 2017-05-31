"""
code: olinda_test.py
description: test suite for olinda
author: JR
IC core team: Jacek Generowicz, JJGC,
G. Martinez, J.A. Hernando, J.M Benlloch
package: invisible cities. See release notes and licence
"""

import os

from pytest import mark, fixture

from   . olinda import Olinda
from   . olinda import OLINDA

@fixture(scope='module')
def conf_file_name(config_tmpdir):
    conf_file_name = str(config_tmpdir.join('olinda.conf'))
    dnn_datafile_name = str(config_tmpdir.join('dnn_olinda_test.h5'))
    weights_file_name = os.path.join(os.environ['ICDIR'],
              'database/test_data/',
              'weights_Kr_FC.h5')

    Olinda.write_config_file(conf_file_name,
                               PATH_OUT = str(config_tmpdir),

                               # currently no way to set these on command line
                               DNN_DATAFILE = dnn_datafile_name,
                               WEIGHTS_FILE = weights_file_name,
                               TEMP_DIR = str(config_tmpdir))
    return conf_file_name

@mark.slow
def test_command_line_olinda(conf_file_name):
    FILE_IN = os.path.join(os.environ['ICDIR'],
              'database/test_data/',
              'kr_ACTIVE_7bar_MCRD.h5')

    # TODO: figure out how to pass extra command line arguments specific to
    #       Olinda
    nrequired, nactual = OLINDA(['OLINDA', '-c', conf_file_name,
                                           '-i', FILE_IN])

    if nrequired > 0:
        assert nrequired == nactual
