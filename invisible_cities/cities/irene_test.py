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
import tables as tb
import numpy  as np
from pytest import mark, fixture

from invisible_cities.cities.irene import Irene, IRENE

@fixture(scope='module')
def conf_file_name_mc(config_tmpdir):
    # Specifies a name for a MC configuration file. Also, default number
    # of events set to 1.

    conf_file_name = str(config_tmpdir.join('irene_mc.conf'))
    Irene.write_config_file(conf_file_name,
                            PATH_OUT = str(config_tmpdir),
                            NEVENTS    = 1)
    return conf_file_name

@fixture(scope='module')
def conf_file_name_data(config_tmpdir):
    # Specifies a name for a data configuration file. Also, default number
    # of events set to 1.

    conf_file_name = str(config_tmpdir.join('irene_data.conf'))
    Irene.write_config_file(conf_file_name,
                            PATH_OUT = str(config_tmpdir),
                            NEVENTS    = 1)
    return conf_file_name


@mark.slow
def test_command_line_irene_electrons_40keV(conf_file_name_mc, config_tmpdir):
    # NB: avoid taking defaults for PATH_IN and PATH_OUT
    # since they are in general test-specific
    # NB: avoid taking defaults for run number (test-specific)

    PATH_IN = os.path.join(os.environ['ICDIR'],
              'database/test_data/',
              'electrons_40keV_z250_RWF.h5')
    PATH_OUT = os.path.join(str(config_tmpdir),
              'electrons_40keV_z250_CWF.h5')

    nrequired, nactual, _ = IRENE(['IRENE',
                                   '-c', conf_file_name_mc,
                                   '-i', PATH_IN,
                                   '-o', PATH_OUT,
                                   '-r', '0'])
    if nrequired > 0:
        assert nrequired == nactual

@mark.slow
def test_command_line_irene_run_2983(conf_file_name_data, config_tmpdir):
    PATH_IN = os.path.join(os.environ['ICDIR'],
              'database/test_data/',
              'run_2983.h5')
    PATH_OUT = os.path.join(str(config_tmpdir),
              'run_2983_pmaps.h5')

    nrequired, nactual, _ = IRENE(['IRENE',
                                   '-c', conf_file_name_data,
                                   '-i', PATH_IN,
                                   '-o', PATH_OUT,
                                   '-r', '2983'])
    if nrequired > 0:
        assert nrequired == nactual

@mark.serial
def test_command_line_irene_runinfo_run_2983(config_tmpdir):
    # Check events numbers & timestamp are copied properly
    PATH_IN = os.path.join(os.environ['ICDIR'],
              'database/test_data/',
              'run_2983.h5')
    PATH_OUT = os.path.join(str(config_tmpdir),
              'run_2983_pmaps.h5')

    with tb.open_file(PATH_IN,  mode='r') as h5in, \
         tb.open_file(PATH_OUT, mode='r') as h5out:
            evts_in  = h5in .root.Run.events[0]
            evts_out = h5out.root.Run.events[0]
            np.testing.assert_array_equal(evts_in, evts_out)

            # Run number is copied
            assert h5in .root.Run.runInfo[0] == h5out.root.Run.runInfo[0]


def test_empty_events_issue_81(conf_file_name_mc, config_tmpdir):
    # NB: explicit PATH_OUT
    PATH_IN = os.path.join(os.environ['ICDIR'],
           'database/test_data/',
           'irene_bug_Kr_ACTIVE_7bar_RWF.h5')
    PATH_OUT = os.path.join(str(config_tmpdir),
              'irene_bug_Kr_ACTIVE_7bar_CWF.h5')

    nrequired, nactual, empty = IRENE(['IRENE',
                                   '-c', conf_file_name_mc,
                                   '-i', PATH_IN,
                                   '-o', PATH_OUT,
                                   '-r', '0'])
    assert nactual == 0
    assert empty == 1
