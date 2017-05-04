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
from collections import namedtuple

from   invisible_cities.cities.irene import Irene, IRENE
from   invisible_cities.reco.pmaps_functions import read_run_and_event_from_pmaps_file

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


@fixture(scope='module')
def job_info_missing_pmts(config_tmpdir, ICDIR):
    # Specifies a name for a data configuration file. Also, default number
    # of events set to 1.
    job_info = namedtuple("job_info",
                          "run_number pmt_missing pmt_active input_filename output_filename")

    run_number  = 3366
    pmt_missing = [11]
    pmt_active  = list(filter(lambda x: x not in pmt_missing, range(12)))


    ifilename   = os.path.join(ICDIR,
                                'database/test_data/',
                                'electrons_40keV_z250_RWF.h5.h5')

    ofilename   = os.path.join(str(config_tmpdir),
                                'electron_40keV_z250_pmaps_missing_PMT.h5')

    return job_info(run_number, pmt_missing, pmt_active, ifilename, ofilename)


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
            nrow = 0
            mctracks_in  = h5in .root.MC.MCTracks[nrow]
            mctracks_out = h5out.root.MC.MCTracks[nrow]
            np.testing.assert_array_equal(mctracks_in, mctracks_out)

            # check events numbers & timestamps
            evts_in  = h5in .root.Run.events[:nactual]
            evts_out = h5out.root.Run.events[:nactual]
            np.testing.assert_array_equal(evts_in, evts_out)



@mark.slow
@mark.serial
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


def test_irene_output_file_structure(conf_file_name_data, config_tmpdir, ICDIR):
    PATH_IN = os.path.join(ICDIR,
                           'database/test_data/',
                           'run_2983.h5')
    PATH_OUT = os.path.join(str(config_tmpdir),
                            'run_2983_pmaps.h5')

    nrequired, nactual, _ = IRENE(['IRENE',
                                   '-c', conf_file_name_data,
                                   '-i', PATH_IN,
                                   '-o', PATH_OUT,
                                   '-n', '3',
                                   '-r', '2983'])

    with tb.open_file(PATH_OUT) as h5out:
        assert "PMAPS"        in h5out.root
        assert "Run"          in h5out.root
        assert "DeconvParams" in h5out.root
        assert "S1"           in h5out.root.PMAPS
        assert "S2"           in h5out.root.PMAPS
        assert "S2Si"         in h5out.root.PMAPS
        assert "events"       in h5out.root.Run
        assert "runInfo"      in h5out.root.Run



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


@mark.slow
def test_irene_electrons_40keV_pmt_active_is_correctly_set(job_info_missing_pmts, config_tmpdir, ICDIR):
    """Run Irene. Write an output file."""

    IRENE = Irene(run_number =  job_info_missing_pmts.run_number,
                  files_in   = [job_info_missing_pmts. input_filename],
                  file_out   =  job_info_missing_pmts.output_filename)

    assert IRENE.pmt_active == job_info_missing_pmts.pmt_active

