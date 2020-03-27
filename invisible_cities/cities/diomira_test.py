import os
import tables as tb
import numpy  as np
import pandas as pd

from pytest import mark
from pytest import raises

from .. core.configure     import                       all as all_events
from .. core.configure     import                 configure
from .. core.testing_utils import    assert_tables_equality
from .. reco               import             tbl_functions as tbl
from .. io  .mcinfo_io     import get_event_numbers_in_file
from .. io  .mcinfo_io     import            load_mchits_df
from .. io  .mcinfo_io     import       load_mcparticles_df

from .  diomira import diomira


def test_diomira_identify_bug(ICDATADIR):
    """Read a one-event file in which the energy of PMTs is equal to zero and
    asset it must be son. This test would fail for a normal file where there
    is always some energy in the PMTs. It's purpose is to provide an automaic
    documentation of a problem (not really a bug but a feature of the
    simulation) that can cause a crash in some events. NEXUS simulation
    can produce eventually events where none of the PMTs of the EP record
    energy. This test simply ensures that the event read in the example file
    in indeed of this type.

    The same event is later processed with Irene (where a protection
    that skips empty events has been added) to ensure that no crash occur."""

    infile = os.path.join(ICDATADIR, 'irene_bug_Kr_ACTIVE_7bar_MCRD.h5')
    with tb.open_file(infile, 'r') as h5in:

        pmtrd  = h5in.root.pmtrd
        pmtwf = pmtrd[0]
        for i in range(pmtrd.shape[1]):
            assert np.sum(pmtwf[i]) == 0


@mark.slow
def test_diomira_copy_mc_and_offset(ICDATADIR, config_tmpdir):
    PATH_IN  = os.path.join(ICDATADIR    , 'electrons_40keV_z250_MCRD.h5')
    PATH_OUT = os.path.join(config_tmpdir, 'electrons_40keV_z250_RWF.h5')

    start_evt  = tbl.event_number_from_input_file_name(PATH_IN)
    run_number = 0
    nrequired = 2

    conf = configure('diomira invisible_cities/config/diomira.conf'.split())

    conf.update(dict(run_number  = run_number,
                     files_in    = PATH_IN,
                     file_out    = PATH_OUT,
                     event_range = (start_evt, start_evt + nrequired)))

    cnt     = diomira(**conf)
    nactual = cnt.events_in
    assert nrequired == nactual

    with tb.open_file(PATH_OUT, mode='r') as h5out:
            # check event & run number
            assert h5out.root.Run.runInfo[0]['run_number'] == run_number
            assert h5out.root.Run.events [0]['evt_number'] == start_evt

            evts_in  = get_event_numbers_in_file(PATH_IN )
            evts_out = get_event_numbers_in_file(PATH_OUT)
            assert len(evts_out) == nrequired
            assert all(evts_in[:nrequired] == evts_out)

            hits_in  = load_mchits_df(PATH_IN )
            hits_out = load_mchits_df(PATH_OUT)
            pd.testing.assert_frame_equal(hits_in.loc[0:nrequired-1],
                                          hits_out                  )


@mark.slow
def test_diomira_mismatch_between_input_and_database(ICDATADIR, output_tmpdir):
    file_in  = os.path.join(ICDATADIR    , 'electrons_40keV_z250_MCRD.h5')
    file_out = os.path.join(output_tmpdir, 'electrons_40keV_z250_RWF_test_mismatch.h5')

    conf = configure('diomira invisible_cities/config/diomira.conf'.split())
    conf.update(dict(run_number  = -4500, # Must be a run number with dead pmts
                     files_in    = file_in,
                     file_out    = file_out,
                     event_range = (0, 1)))
    conf["trigger_params"].update(dict(tr_channels = (18, 19)))

    cnt = diomira(**conf)

    # we are just interested in checking whether the code runs or not
    assert cnt.events_in == 1


@mark.slow
def test_diomira_trigger_on_masked_pmt_raises_ValueError(ICDATADIR, output_tmpdir):
    file_in  = os.path.join(ICDATADIR    , 'electrons_40keV_z250_MCRD.h5')
    file_out = os.path.join(output_tmpdir, 'electrons_40keV_z250_RWF_test_trigger.h5')

    conf = configure('diomira invisible_cities/config/diomira.conf'.split())
    conf.update(dict(run_number     = -4500, # Must be a run number with dead pmts
                     files_in       = file_in,
                     file_out       = file_out,
                     trigger_type   = "S2",
                     trigger_params = dict(
                        tr_channels         = (0,),  # This is a masked PMT for this run
                        min_number_channels = 1   ,
                        data_mc_ratio       = 1   ,
                        min_height          = 0   ,
                        max_height          = 1e6 ,
                        min_charge          = 0   ,
                        max_charge          = 1e6 ,
                        min_width           = 0   ,
                        max_width           = 1e6 ),
                     event_range  = (0, 1)))

    with raises(ValueError):
        diomira(**conf)

def test_diomira_read_multiple_files(ICDATADIR, output_tmpdir):
    file_in     = os.path.join(ICDATADIR                                       ,
                               "Kr83_nexus_v5_03_00_ACTIVE_7bar_3evts*.MCRD.h5")
    file_out    = os.path.join(output_tmpdir                                 ,
                               "Kr83_nexus_v5_03_00_ACTIVE_7bar_6evts.RWF.h5")

    nevents_per_file = 3

    nrequired = 10
    conf = configure('dummy invisible_cities/config/diomira.conf'.split())
    conf.update(dict(run_number   = -4734,
                     files_in     = file_in,
                     file_out     = file_out,
                     event_range  = (0, nrequired)))
    conf["trigger_params"].update(dict(tr_channels = [18, 19]))

    diomira(**conf)

    first_file  = os.path.join(ICDATADIR                                      ,
                               "Kr83_nexus_v5_03_00_ACTIVE_7bar_3evts.MCRD.h5")
    second_file = os.path.join(ICDATADIR                                       ,
                               "Kr83_nexus_v5_03_00_ACTIVE_7bar_3evts_1.MCRD.h5")

    particles_in1 = load_mcparticles_df( first_file)
    hits_in1      = load_mchits_df     ( first_file)
    particles_in2 = load_mcparticles_df(second_file)
    hits_in2      = load_mchits_df     (second_file)
    particles_out = load_mcparticles_df(   file_out)
    hits_out      = load_mchits_df     (   file_out)

    evt_in  = np.concatenate([hits_in1.index.levels[0],
                              hits_in2.index.levels[0]])
    evt_out = hits_out.index.levels[0]
    assert all(evt_in == evt_out)

    all_hit_in      = pd.concat([hits_in1     ,      hits_in2])
    pd.testing.assert_frame_equal(all_hit_in, hits_out)
    all_particle_in = pd.concat([particles_in1, particles_in2])
    pd.testing.assert_frame_equal(all_particle_in, particles_out)


def test_diomira_exact_result(ICDATADIR, output_tmpdir):
    file_in     = os.path.join(ICDATADIR                                      ,
                               "Kr83_nexus_v5_03_00_ACTIVE_7bar_3evts.MCRD.h5")
    file_out    = os.path.join(output_tmpdir                                  ,
                               "exact_result_diomira.h5"                      )
    true_output = os.path.join(ICDATADIR                                      ,
                               "Kr83_nexus_v5_03_00_ACTIVE_7bar_3evts.RWF.h5" )

    conf = configure("diomira invisible_cities/config/diomira.conf".split())
    conf.update(dict(run_number   = -6340,
                     files_in     = file_in,
                     file_out     = file_out,
                     trigger_type = None,
                     event_range  = all_events))

    # Set a specific seed because we want the result to be
    # repeatible. Go back to original state after running.
    original_random_state = np.random.get_state()
    np.random.seed(123456789)
    diomira(**conf)
    np.random.set_state(original_random_state)

    ## tables = (     "MC/extents",  "MC/hits"   , "MC/particles", "MC/generators",
    ##                "RD/pmtrwf" ,  "RD/pmtblr" , "RD/sipmrwf"  ,
    ##               "Run/events" , "Run/runInfo",
    ##           "Filters/trigger")
    tables = (     "RD/pmtrwf" ,  "RD/pmtblr" , "RD/sipmrwf"  ,
                  "Run/events" , "Run/runInfo",
              "Filters/trigger")
    with tb.open_file(true_output)  as true_output_file:
        with tb.open_file(file_out) as      output_file:
            for table in tables:
                got      = getattr(     output_file.root, table)
                expected = getattr(true_output_file.root, table)
                assert_tables_equality(got, expected)


def test_diomira_can_fix_random_seed(output_tmpdir):
    file_out    = os.path.join(output_tmpdir, "exact_result_diomira.h5")

    conf = configure("diomira invisible_cities/config/diomira.conf".split())
    conf.update(dict(random_seed = 123),
                file_out     = file_out)
    diomira(**conf)
