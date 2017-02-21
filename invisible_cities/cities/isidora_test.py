"""
code: isidora_test.py
description: test suite for isidora
author: J.J. Gomez-Cadenas
IC core team: Jacek Generowicz, JJGC,
G. Martinez, J.A. Hernando, J.M Benlloch
package: invisible cities. See release notes and licence
last changed: 01-12-2017
"""

from pytest import mark, raises, fixture

from   invisible_cities.core.exceptions import NoInputFiles
from   invisible_cities.cities.isidora import Isidora, ISIDORA


@fixture
def conf_file_name(config_tmpdir):
    conf_file_name = str(config_tmpdir.join('maurilia.conf'))
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
def test_command_line_isidora(conf_file_name):
    nrequired, nactual = ISIDORA(['ISIDORA', '-c', conf_file_name])
    if nrequired > 0:
        assert nrequired == nactual
