"""
code: irene_test.py
description: test suite for irene
author: J.J. Gomez-Cadenas
IC core team: Jacek Generowicz, JJGC,
G. Martinez, J.A. Hernando, J.M Benlloch
package: invisible cities. See release notes and licence
last changed: 01-12-2017
"""
import os
from pytest import mark, fixture

from   invisible_cities.cities.irene import Irene, IRENE

@fixture
def conf_file_name(config_tmpdir):
    conf_file_name = str(config_tmpdir.join('maurilia.conf'))
    Irene.write_config_file(conf_file_name,
                            PATH_OUT = str(config_tmpdir))
    return conf_file_name


@mark.slow
def test_command_line_irene(conf_file_name):
    nrequired, nactual, _ = IRENE(['IRENE', '-c', conf_file_name])
    if nrequired > 0:
        assert nrequired == nactual

def test_empty_events_issue_81(conf_file_name):
    PATH_IN = os.path.join(os.environ['ICDIR'],
           'database/test_data/',
           'irene_bug_Kr_ACTIVE_7bar_RWF.h5')

    nrequired, nactual, empty = IRENE(['IRENE', '-c', conf_file_name, '-i', PATH_IN])
    assert nactual == 0
    assert empty == 1
