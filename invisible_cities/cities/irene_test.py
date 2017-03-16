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

from   invisible_cities.cities.irene import Irene, IRENE
from   invisible_cities.core.system_of_units_c import units
from   invisible_cities.reco.params import S12Params as S12P
from   invisible_cities.reco.params import SensorParams
import invisible_cities.reco.tbl_functions as tbl
from   invisible_cities.reco.pmaps_functions import (
    read_pmaps, read_run_and_event_from_pmaps_file)

@fixture(scope='module')
def conf_file_name_mc(config_tmpdir):
    # Specifies a name for a MC configuration file. Also, default number
    # of events set to 1.

    conf_file_name = str(config_tmpdir.join('irene_mc.conf'))
    Irene.write_config_file(conf_file_name,
                            PATH_OUT = str(config_tmpdir),
                            NEVENTS  = 1)
    return conf_file_name

@fixture(scope='module')
def conf_file_name_data(config_tmpdir):
    # Specifies a name for a data configuration file. Also, default number
    # of events set to 1.

    conf_file_name = str(config_tmpdir.join('irene_data.conf'))
    Irene.write_config_file(conf_file_name,
                            PATH_OUT = str(config_tmpdir),
                            NEVENTS  = 1)
    return conf_file_name


@mark.slow
def test_command_line_irene_electrons_40keV(conf_file_name_mc, config_tmpdir, ICDIR):
    # NB: avoid taking defaults for PATH_IN and PATH_OUT
    # since they are in general test-specific
    # NB: avoid taking defaults for run number (test-specific)

    PATH_IN = os.path.join(ICDIR,
                           'database/test_data/',
                           'electrons_40keV_z250_RWF.h5')
    PATH_OUT = os.path.join(str(config_tmpdir),
                            'electrons_40keV_z250_CWF.h5')

    nrequired, nactual, _ = IRENE(['IRENE',
                                   '-c', conf_file_name_mc,
                                   '-i', PATH_IN,
                                   '-o', PATH_OUT,
                                   '-n', '3',
                                   '-r', '0'])
    if nrequired > 0:
        assert nrequired == nactual

    with tb.open_file(PATH_IN,  mode='r') as h5in, \
         tb.open_file(PATH_OUT, mode='r') as h5out:
            mctracks_in  = h5in .root.MC.MCTracks[0]
            mctracks_out = h5out.root.MC.MCTracks[0]
            np.testing.assert_array_equal(mctracks_in, mctracks_out)


@mark.slow
def test_command_line_irene_run_2983(conf_file_name_data, config_tmpdir, ICDIR):
    """Run Irene. Write an output file."""

    # NB: the input file has 5 events. The maximum value for 'n'
    # in the IRENE parameters is 5, but it can run with a smaller values
    # (eg, 2) to speed the test.

    PATH_IN = os.path.join(ICDIR,
                           'database/test_data/',
                           'run_2983.h5')
    PATH_OUT = os.path.join(str(config_tmpdir),
                            'run_2983_pmaps.h5')

    nrequired, nactual, _ = IRENE(['IRENE',
                                   '-c', conf_file_name_data,
                                   '-i', PATH_IN,
                                   '-o', PATH_OUT,
                                   '-n','2',
                                   '-r', '2983'])
    if nrequired > 0:
        assert nrequired == nactual

@mark.serial
def test_command_line_irene_runinfo_run_2983(config_tmpdir, ICDIR):
    """Read back the file written by previous test. Check runinfo."""

    # NB: the input file has 5 events. The maximum value for 'n'
    # in the IRENE parameters is 5, but it can run with a smaller values
    # (eg, 2) to speed the test. BUT NB, this has to be propagated to this
    # test, eg. h5in .root.Run.events[0:2] if one has run 2 events.

    #import pdb; pdb.set_trace()
    PATH_IN = os.path.join(ICDIR,
                           'database/test_data/',
                           'run_2983.h5')
    PATH_OUT = os.path.join(str(config_tmpdir),
                            'run_2983_pmaps.h5')

    with tb.open_file(PATH_IN, mode='r') as h5in:
        evt_in  = h5in.root.Run.events[0:2]
        evts_in = []
        ts_in   = []
        for e in evt_in:
            evts_in.append(e[0])
            ts_in  .append(e[1])

        rundf, evtdf = read_run_and_event_from_pmaps_file(PATH_OUT)
        evts_out = evtdf.evt_number.values
        ts_out = evtdf.timestamp.values
        np.testing.assert_array_equal(evts_in, evts_out)
        np.testing.assert_array_equal(  ts_in,   ts_out)

        rin = h5in.root.Run.runInfo[:][0][0]
        rout = rundf.run_number[0]
        assert rin == rout


def test_empty_events_issue_81(conf_file_name_mc, config_tmpdir, ICDIR):
    # NB: explicit PATH_OUT
    PATH_IN = os.path.join(ICDIR,
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
    assert empty   == 1
