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
from   invisible_cities.core.params import S12Params as S12P
from   invisible_cities.core.params import SensorParams
from invisible_cities.core.system_of_units_c import units
import invisible_cities.core.tbl_functions as tbl

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

@mark.slow
def test_pmaps_store_issue_151(config_tmpdir):
    # Check that PMAPS tables are written correctly
    PATH_IN = os.path.join(os.environ['ICDIR'],
              'database/test_data/',
              'run_2983.h5')
    PATH_OUT = os.path.join(str(config_tmpdir),
              'run_2983_pmaps.h5')

    s1par = S12P(tmin   = 10 * units.mus,
                 tmax   = 590 * units.mus,
                 stride = 4,
                 lmin   = 6,
                 lmax   = 16,
                 rebin  = False)

    s2par = S12P(tmin   = 590 * units.mus,
                 tmax   = 620 * units.mus,
                 stride = 40,
                 lmin   = 100,
                 lmax   = 1000,
                 rebin  = True)

    irene = Irene(run_number  = 2983,
                 files_in    = [PATH_IN],
                 file_out    = PATH_OUT,
                 compression = 'ZLIB4',
                 nprint      = 1,
                 n_baseline  = 38000,
                 thr_trigger = 5.0 * units.adc,
                 n_MAU       = 100,
                 thr_MAU     = 3.0 * units.adc,
                 thr_csum_s1 = 0.2 * units.adc,
                 thr_csum_s2 = 1.0 * units.adc,
                 n_MAU_sipm  = 100,
                 thr_sipm    = 5.0 * units.pes,
                 s1_params   = s1par,
                 s2_params   = s2par,
                 thr_sipm_s2 = 30. * units.pes)

    with tb.open_file(irene.input_files[0], "r") as h5in:
        with tb.open_file(irene.output_file, "w",
             filters = tbl.filters(irene.compression)) as pmap_file:

             irene._set_pmap_store(pmap_file)
             irene.eventsInfo = h5in.root.Run.events
             pmtrwf  = h5in.root.RD. pmtrwf
             sipmrwf = h5in.root.RD.sipmrwf

             CWF = irene.deconv_pmt(pmtrwf[0])
             csum, csum_mau = irene.calibrated_pmt_sum(CWF)
             s1_ene, s1_indx = irene.csum_zs(csum_mau,
                                             threshold = irene.thr_csum_s1)
             s2_ene, s2_indx = irene.csum_zs(csum,
                                             threshold = irene.thr_csum_s2)

             sipmzs = irene.calibrated_signal_sipm(sipmrwf[0])
             S1, S2 = irene.find_S12(s1_ene, s1_indx, s2_ene, s2_indx)
             S2Si = irene.find_S2Si(S2, sipmzs)
             irene._store_pmaps(0, 0, S1, S2, S2Si)

             # number of rows in file for one event must be the same than
             # length of time (or energy) array for S1/S2
             irene.s1t.flush()
             irene.s2t.flush()
             irene.s2sit.flush()

             assert irene.s1t.shape[0] == len(S1[0][0])
             assert irene.s2t.shape[0] == len(S2[0][0])
             assert irene.s2sit.shape[0] > 0
