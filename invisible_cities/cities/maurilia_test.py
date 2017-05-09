"""
code: maurilia_test.py
description: test suite for maurilia
author: Josh Renner
IC core team: Jacek Generowicz, JJGC,
G. Martinez, J.A. Hernando, J.M Benlloch
package: invisible cities. See release notes and licence
"""
from pytest import mark, fixture
from   invisible_cities.cities.maurilia import Maurilia, MAURILIA


@fixture(scope='module')
def conf_file_name(config_tmpdir):
    conf_file_name = str(config_tmpdir.join('maurilia.conf'))
    Maurilia.write_config_file(conf_file_name,
                               FILE_IN  = 'NEW_se_mc_reco_pmaps_1evt.h5',
                               FILE_OUT = 'NEW_se_mc_1evt.h5',
                               PATH_OUT = str(config_tmpdir))
    return conf_file_name


def test_command_line_maurilia(conf_file_name):
    MAURILIA(['MAURILIA', '-c', conf_file_name])
