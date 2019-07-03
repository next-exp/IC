import os

from pytest import mark

import numpy  as np
import tables as tb

from .. io.dst_io            import load_dst
from .. core.testing_utils   import assert_dataframes_close
from .. core.testing_utils   import assert_tables_equality
from .. core.configure       import configure
from .. core.configure       import all as all_events
from .. core.system_of_units import pes, mm, mus, ns

from .  dorothea import dorothea


def test_dorothea_KrMC(config_tmpdir, KrMC_pmaps_filename, KrMC_kdst):
    # NB: avoid taking defaults for PATH_IN and PATH_OUT
    # since they are in general test-specific
    # NB: avoid taking defaults for run number (test-specific)

    PATH_IN   = KrMC_pmaps_filename
    PATH_OUT  = os.path.join(config_tmpdir, 'KrDST.h5')
    nrequired = 10
    df_true   = KrMC_kdst[0].true

    conf = configure('dummy invisible_cities/config/dorothea.conf'.split())
    conf.update(dict(files_in    = PATH_IN,
                     file_out    = PATH_OUT,
                     event_range = (0, nrequired),
                     **KrMC_kdst[0].config))


    cnt      = dorothea(**conf)
    nevt_in  = cnt.events_in
    nevt_out = cnt.events_out

    if nrequired > 0:
        assert nrequired == nevt_in
        assert nevt_out  == len(df_true)

    dst = load_dst(PATH_OUT,
                   group = "DST",
                   node  = "Events")
    assert len(set(dst.event)) == nevt_out

    assert_dataframes_close(dst, df_true, False, rtol=1e-2)


def test_dorothea_filter_events(config_tmpdir, Kr_pmaps_run4628_filename):
    PATH_IN =  Kr_pmaps_run4628_filename

    PATH_OUT = os.path.join(config_tmpdir, 'KrDST_4628.h5')
    nrequired = 50
    conf = configure('dummy invisible_cities/config/dorothea.conf'.split())
    conf.update(dict(run_number  = 4628,
                     files_in    = PATH_IN,
                     file_out    = PATH_OUT,

                     drift_v     =      2 * mm / mus,
                     s1_nmin     =      1,
                     s1_nmax     =      1,
                     s1_emin     =      1 * pes,
                     s1_emax     =     30 * pes,
                     s1_wmin     =    100 * ns,
                     s1_wmax     =    300 * ns,
                     s1_hmin     =      1 * pes,
                     s1_hmax     =      5 * pes,
                     s1_ethr     =    0.5 * pes,
                     s2_nmin     =      1,
                     s2_nmax     =      2,
                     s2_emin     =    1e3 * pes,
                     s2_emax     =    1e4 * pes,
                     s2_wmin     =      2 * mus,
                     s2_wmax     =     20 * mus,
                     s2_hmin     =    1e3 * pes,
                     s2_hmax     =    1e5 * pes,
                     s2_ethr     =      1 * pes,
                     s2_nsipmmin =      5,
                     s2_nsipmmax =     30,
                     event_range = (0, nrequired)))

    events_pass = [ 1,  4, 10, 19, 20, 21, 26,
                   26, 29, 33, 41, 43, 45, 46]
    s1_peak_pass = [ 0,  0,  0,  0,  0,  0,  0,
                     0,  0,  0,  0,  0,  0,  0]
    s2_peak_pass = [ 0,  0,  0,  0,  0,  0,  0,
                     1,  0,  0,  0,  0,  0,  0]

    cnt = dorothea(**conf)

    nevt_in  = cnt.events_in
    nevt_out = cnt.events_out
    assert nrequired    == nevt_in
    assert nevt_out     == len(set(events_pass))

    dst = load_dst(PATH_OUT, "DST", "Events")
    assert len(set(dst.event.values)) == nevt_out

    assert np.all(dst.event  .values ==  events_pass)
    assert np.all(dst.s1_peak.values == s1_peak_pass)
    assert np.all(dst.s2_peak.values == s2_peak_pass)



@mark.skip(reason="The configuration of corona in dorothea can only return 1 cluster."
                  "The assertion of more than one cluster has been removed")
def test_dorothea_issue_347(Kr_pmaps_run4628_filename, config_tmpdir):
    PATH_IN =  Kr_pmaps_run4628_filename
    PATH_OUT = os.path.join(config_tmpdir, 'KrDST.h5')
    conf = configure('dummy invisible_cities/config/dorothea_with_corona.conf'.split())

    # with this parameters Corona will find several clusters
    conf.update(dict(run_number    = 4628,
                     files_in      = PATH_IN,
                     file_out      = PATH_OUT,
                     lm_radius     = 10.0,
                     new_lm_radius = 13.0,
                     msipm         = 1))
    cnt = dorothea(**conf)
    assert cnt.n_events_more_than_1_cluster == 3


@mark.skip("This scenario is not possible in liquid cities")
def test_dorothea_event_not_found(ICDATADIR, output_tmpdir):
    file_in   = os.path.join(ICDATADIR    , "kr_rwf_0_0_7bar_NEXT_v1_00_05_v0.9.2_20171011_krmc_irene_3evt.h5")
    file_out  = os.path.join(output_tmpdir, "test_dorothea_event_not_found.h5")

    conf = configure('dummy invisible_cities/config/dorothea.conf'.split())
    nevt = 3

    conf.update(dict(files_in    = file_in,
                     file_out    = file_out,
                     event_range = (0, nevt)))

    cnt = dorothea(**conf)
    assert cnt.n_empty_pmaps == 1


def test_dorothea_exact_result(ICDATADIR, output_tmpdir):
    file_in     = os.path.join(ICDATADIR    ,  "Kr83_nexus_v5_03_00_ACTIVE_7bar_3evts.PMP.h5")
    file_out    = os.path.join(output_tmpdir,                      "exact_result_dorothea.h5")
    true_output = os.path.join(ICDATADIR    , "Kr83_nexus_v5_03_00_ACTIVE_7bar_3evts.KDST.h5")

    conf = configure("dorothea invisible_cities/config/dorothea.conf".split())
    conf.update(dict(run_number   = -6340,
                     files_in     = file_in,
                     file_out     = file_out,
                     event_range  = all_events))

    dorothea(**conf)

    tables = (    "DST/Events",
              "Filters/s12_selector")
    with tb.open_file(true_output)  as true_output_file:
        with tb.open_file(file_out) as      output_file:
            for table in tables:
                got      = getattr(     output_file.root, table)
                expected = getattr(true_output_file.root, table)
                assert_tables_equality(got, expected)


def test_dorothea_empty_input_file(config_tmpdir, ICDATADIR):
    # Dorothea must run on an empty file without raising any exception
    # The input file has the complete structure of a PMAP but no events.

    PATH_IN  = os.path.join(ICDATADIR    , 'empty_pmaps.h5')
    PATH_OUT = os.path.join(config_tmpdir, 'empty_kdst.h5')

    conf = configure('dummy invisible_cities/config/dorothea.conf'.split())
    conf.update(dict(files_in      = PATH_IN,
                     file_out      = PATH_OUT))

    dorothea(**conf)
