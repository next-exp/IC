import os

from pytest import mark
from pytest import fixture

from . zaira import Zaira
from . zaira import ZAIRA


@fixture(scope='module')
def conf_file_name_mc(config_tmpdir):
    # Specifies a name for a MC configuration file. Also, default number
    # of events set to 1.

    conf_file_name = str(config_tmpdir.join('zaira_mc.conf'))
    Zaira.write_config_file(conf_file_name,
                            PATH_OUT = str(config_tmpdir))
    return conf_file_name


@mark.slow
def test_command_line_zaira_KrMC(conf_file_name_mc, config_tmpdir, ICDIR):
    # NB: avoid taking defaults for PATH_IN and PATH_OUT
    # since they are in general test-specific
    # NB: avoid taking defaults for run number (test-specific)

    PATH_IN =  os.path.join(ICDIR,
                            'database/test_data', 'KrDST_MC.h5')
    PATH_OUT = os.path.join(str(config_tmpdir), 'KrCorr.h5')

    nevt = ZAIRA(['Zaira',
                  '-c', conf_file_name_mc,
                  '-i', PATH_IN,
                  '-o', PATH_OUT,
                  '-r', '0'])

    assert nevt > 0
