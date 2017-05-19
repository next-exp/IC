"""
code: isidora_test.py
description: test suite for isidora
author: J.J. Gomez-Cadenas
IC core team: Jacek Generowicz, JJGC,
G. Martinez, J.A. Hernando, J.M Benlloch
package: invisible cities. See release notes and licence
last changed: 01-12-2017
"""
import os

from   pytest import mark
from   pytest import raises
from   pytest import fixture

from .. core.exceptions import NoInputFiles
from .  isidora import Isidora
from .  isidora import ISIDORA


@fixture(scope='module')
def conf_file_name(config_tmpdir):
    conf_file_name = str(config_tmpdir.join('isidora.conf'))
    Isidora.write_config_file(conf_file_name,
                              PATH_OUT = str(config_tmpdir))
    return conf_file_name

def test_isidora_no_input():
    """Test that Isidora throws NoInputFiles exceptions if no input files
       are defined. """

    with raises(NoInputFiles):
        fpp = Isidora(run_number = 0)
        fpp.run(nmax=1)


@mark.slow
def test_command_line_isidora(conf_file_name, config_tmpdir):
    PATH_IN = os.path.join(os.environ['ICDIR'],
           'database/test_data/',
           'electrons_40keV_z250_RWF.h5')
    PATH_OUT = os.path.join(str(config_tmpdir),
              'electrons_40keV_z250_CWF.h5')

    nrequired, nactual = ISIDORA(['ISIDORA', '-c', conf_file_name,
                                  '-c', conf_file_name,
                                  '-i', PATH_IN,
                                  '-o', PATH_OUT,
                                  '-n', '2',
                                  '-r', '0'])
    if nrequired > 0:
        assert nrequired == nactual
