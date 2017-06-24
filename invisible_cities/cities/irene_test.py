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
from .. core.configure       import configure
from .. core.ic_types        import minmax
from .. reco.pmaps_functions import read_run_and_event_from_pmaps_file
from .. reco.params          import S12Params as S12P


@fixture(scope='module')
def s12params():
    s1par = S12P(time = minmax(min   =  99 * units.mus,
                               max   = 101 * units.mus),
                 length = minmax(min =   4,
                                 max =  20,),
                 stride              =   4,
                 rebin               = False)

    s2par = S12P(time = minmax(min   =    101 * units.mus,
                               max   =   1199 * units.mus),
                 length = minmax(min =     80,
                                 max = 200000),
                 stride              =     40,
                 rebin               = True)
    return s1par, s2par


def unpack_s12params(s12params):
    s1par, s2par = s12params
    return dict(s1_tmin   = s1par.time.min,
                s1_tmax   = s1par.time.max,
                s1_string = s1par.stride,
                s1_rebin  = s1par.rebin,
                s1_lmin   = s1par.length.min,
                s1_lmax   = s1par.length.max,

                s2_tmin   = s2par.time.min,
                s2_tmax   = s2par.time.max,
                s2_string = s2par.stride,
                s2_rebin  = s2par.rebin,
                s2_lmin   = s2par.length.min,
                s2_lmax   = s2par.length.max)


@fixture(scope='module')
def job_info_missing_pmts(config_tmpdir, ICDIR):
    # Specifies a name for a data configuration file. Also, default number
    # of events set to 1.
    job_info = namedtuple("job_info",
                          "run_number pmt_missing pmt_active input_filename output_filename")

    run_number  = 3366
    pmt_missing = [11]
    pmt_active  = list(filter(lambda x: x not in pmt_missing, range(12)))


    ifilename   = os.path.join(ICDIR, 'database/test_data/', 'electrons_40keV_z250_RWF.h5.h5')
    ofilename   = os.path.join(config_tmpdir,                'electrons_40keV_z250_pmaps_missing_PMT.h5')

    return job_info(run_number, pmt_missing, pmt_active, ifilename, ofilename)


@mark.slow
def test_irene_electrons_40keV(config_tmpdir, ICDIR, s12params):
    # NB: avoid taking defaults for PATH_IN and PATH_OUT
    # since they are in general test-specific
    # NB: avoid taking defaults for run number (test-specific)

    PATH_IN = os.path.join(ICDIR, 'database/test_data/', 'electrons_40keV_z250_RWF.h5')
    PATH_OUT = os.path.join(config_tmpdir,               'electrons_40keV_z250_CWF.h5')

    nrequired  = 2

    conf = configure('dummy invisible_cities/config/irene.conf'.split()).as_dict
    conf.update(dict(run_number = 0,
                     filesin   = PATH_IN,
                     file_out   = PATH_OUT,
                     nmax       = nrequired,
                     **unpack_s12params(s12params)))

    irene = Irene(**conf)

    nactual, _ = irene.run()
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

    PATH_IN  = os.path.join(ICDIR, 'database/test_data/', 'run_2983.h5')
    PATH_OUT = os.path.join(config_tmpdir, 'run_2983_pmaps.h5')

    nrequired = 2

    conf = configure('dummy invisible_cities/config/irene.conf'.split()).as_dict
    conf.update(dict(run_number = 2983,
                     files_in   = PATH_IN,
                     file_out   = PATH_OUT,
                     nmax       = nrequired,
                     **unpack_s12params(s12params)))

    irene = Irene(**conf)

    nactual, _ = irene.run()
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

<<<<<<< HEAD
    PATH_IN = os.path.join(ICDIR, 'database/test_data/', 'run_2983.h5')
    PATH_OUT = os.path.join(config_tmpdir,               'run_2983_pmaps.h5')
=======
    PATH_IN = os.path.join(ICDIR,
                           'database/test_data/',
                           'run_2983.h5')
    PATH_OUT = os.path.join(str(config_tmpdir),
                            'run_2983_pmaps.h5')
>>>>>>> Adapt test_irene_40keV...correctly_set to new structure

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
    PATH_OUT = os.path.join(config_tmpdir, 'run_2983_pmaps.h5')

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
    PATH_IN = os.path.join(ICDIR, 'database/test_data/', 'irene_bug_Kr_ACTIVE_7bar_RWF.h5')
    PATH_OUT = os.path.join(config_tmpdir,               'irene_bug_Kr_ACTIVE_7bar_CWF.h5')

    nrequired = 10

    conf = configure('dummy invisible_cities/config/irene.conf'.split()).as_dict
    conf.update(dict(run_number = 0,
                     files_in   = PATH_IN,
                     file_out   = PATH_OUT,
                     nmax       = nrequired,
                     **unpack_s12params(s12params)))

    irene = Irene(**conf)

    nactual, nempty = irene.run()

    assert nactual == 0
    assert nempty  == 1


def test_irene_electrons_40keV_pmt_active_is_correctly_set(job_info_missing_pmts, config_tmpdir, ICDIR, s12params):
    "Check that PMT active correctly describes the PMT configuration of the detector"
    nrequired = 1
    conf = configure('dummy invisible_cities/config/irene.conf'.split()).as_dict
    conf.update(dict(run_number =  job_info_missing_pmts.run_number,
                     files_in   =  job_info_missing_pmts. input_filename,
                     file_out   =  job_info_missing_pmts.output_filename,
                     nmax       = nrequired,
                     **unpack_s12params(s12params))) # s12params are just dummy values in this test

    irene = Irene(**conf)

    assert irene.pmt_active == job_info_missing_pmts.pmt_active
