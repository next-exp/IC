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
from collections import namedtuple

import tables as tb
import numpy  as np

from pytest import mark
from pytest import fixture

from .  irene import Irene
from .. core                 import system_of_units as units
from .. reco.pmaps_functions import read_run_and_event_from_pmaps_file
from .. reco.params          import S12Params as S12P


@fixture(scope='module')
def s12params():
    s1par = S12P(tmin   =  99 * units.mus,
                 tmax   = 101 * units.mus,
                 lmin   =   4,
                 lmax   =  20,
                 stride =   4,
                 rebin  = False)

    s2par = S12P(tmin   =    101 * units.mus,
                 tmax   =   1199 * units.mus,
                 lmin   =     80,
                 lmax   = 200000,
                 stride =     40,
                 rebin  = True)
    return s1par, s2par


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
def test_irene_electrons_40keV(config_tmpdir, ICDIR, s12params):
    # NB: avoid taking defaults for PATH_IN and PATH_OUT
    # since they are in general test-specific
    # NB: avoid taking defaults for run number (test-specific)

    PATH_IN = os.path.join(ICDIR,
                           'database/test_data/',
                           'electrons_40keV_z250_RWF.h5')
    PATH_OUT = os.path.join(str(config_tmpdir),
                            'electrons_40keV_z250_CWF.h5')

    s1par, s2par = s12params

    irene = Irene(run_number = 0,
                  files_in   = [PATH_IN],
                  file_out   = PATH_OUT,
                  s1_params  = s1par,
                  s2_params  = s2par)

    nrequired  = 2
    nactual, _ = irene.run(nmax = nrequired)
    if nrequired > 0:
        assert nrequired == nactual

    with tb.open_file(PATH_IN,  mode='r') as h5in, \
         tb.open_file(PATH_OUT, mode='r') as h5out:
            nrow = 0
            mctracks_in  = h5in .root.MC.MCTracks[nrow]
            mctracks_out = h5out.root.MC.MCTracks[nrow]
            np.testing.assert_array_equal(mctracks_in, mctracks_out)

            # check events numbers & timestamps
            evts_in     = h5in .root.Run.events[:nactual]
            evts_out_u8 = h5out.root.Run.events[:nactual]
            # The old format used <i4 for th event number; the new one
            # uses <u8. Casting the latter to the former allows us to
            # re-use the old test data files.
            evts_out_i4 = evts_out_u8.astype([('evt_number', '<i4'), ('timestamp', '<u8')])
            np.testing.assert_array_equal(evts_in, evts_out_i4)


@mark.slow
@mark.serial
def test_irene_run_2983(config_tmpdir, ICDIR, s12params):
    """Run Irene. Write an output file."""

    # NB: the input file has 5 events. The maximum value for 'n'
    # in the IRENE parameters is 5, but it can run with a smaller values
    # (eg, 2) to speed the test.

    PATH_IN = os.path.join(ICDIR,
                           'database/test_data/',
                           'run_2983.h5')
    PATH_OUT = os.path.join(str(config_tmpdir),
                            'run_2983_pmaps.h5')

    s1par, s2par = s12params

    irene = Irene(run_number = 2983,
                  files_in   = [PATH_IN],
                  file_out   = PATH_OUT,
                  s1_params  = s1par,
                  s2_params  = s2par)

    nrequired  = 2
    nactual, _ = irene.run(nmax = nrequired)
    if nrequired > 0:
        assert nrequired == nactual


@mark.slow # not slow itself, but depends on a slow test
@mark.serial
def test_irene_runinfo_run_2983(config_tmpdir, ICDIR):
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


@mark.serial
@mark.slow
def test_irene_output_file_structure(config_tmpdir, ICDIR):
    PATH_OUT = os.path.join(str(config_tmpdir), 'run_2983_pmaps.h5')

    with tb.open_file(PATH_OUT) as h5out:
        assert "PMAPS"        in h5out.root
        assert "Run"          in h5out.root
        assert "DeconvParams" in h5out.root
        assert "S1"           in h5out.root.PMAPS
        assert "S2"           in h5out.root.PMAPS
        assert "S2Si"         in h5out.root.PMAPS
        assert "events"       in h5out.root.Run
        assert "runInfo"      in h5out.root.Run


def test_empty_events_issue_81(config_tmpdir, ICDIR, s12params):
    # NB: explicit PATH_OUT
    PATH_IN = os.path.join(ICDIR,
                           'database/test_data/',
                           'irene_bug_Kr_ACTIVE_7bar_RWF.h5')
    PATH_OUT = os.path.join(str(config_tmpdir),
                            'irene_bug_Kr_ACTIVE_7bar_CWF.h5')

    s1par, s2par = s12params

    irene = Irene(run_number = 0,
                  files_in   = [PATH_IN],
                  file_out   = PATH_OUT,
                  s1_params  = s1par,
                  s2_params  = s2par)

    nactual, nempty = irene.run(nmax = 10)

    assert nactual == 0
    assert nempty  == 1


def test_irene_electrons_40keV_pmt_active_is_correctly_set(job_info_missing_pmts, config_tmpdir, ICDIR):
    "Check that PMT active correctly describes the PMT configuration of the detector"
    irene = Irene(run_number =  job_info_missing_pmts.run_number,
                  files_in   = [job_info_missing_pmts. input_filename],
                  file_out   =  job_info_missing_pmts.output_filename,
                  s1_params  = S12P('dummy','not','used','in','the','test'),
                  s2_params  = S12P('dummy','not','used','in','the','test'))

    assert irene.pmt_active == job_info_missing_pmts.pmt_active
