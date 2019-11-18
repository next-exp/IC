import os
import tables as tb
import numpy  as np

from pytest import mark
from pytest import raises

from .. core                 import system_of_units as units
from .. core.configure       import             all as all_events
from .. core.configure       import configure
from .. core.testing_utils   import assert_tables_equality

from .. reco     import tbl_functions as tbl
from .. sierpe   import fee as FEE

from .  diomira  import diomira


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

    with tb.open_file(PATH_IN,  mode='r') as h5in, \
         tb.open_file(PATH_OUT, mode='r') as h5out:
            # check event & run number
            assert h5out.root.Run.runInfo[0]['run_number'] == run_number
            assert h5out.root.Run.events [0]['evt_number'] == start_evt

            # check mcextents
            # we have to convert manually into a tuple because MCTracks[0]
            # returns an object of type numpy.void where we cannot index
            # using ranges like mctracks_in[1:]
            mcextents_in  = tuple(h5in .root.MC.extents[0])
            mcextents_out = tuple(h5out.root.MC.extents[0])
            #evt number is not equal if we redefine first event number
            assert mcextents_out[0] == start_evt
            for e in zip(mcextents_in[1:], mcextents_out[1:]):
                np.testing.assert_array_equal(e[0],e[1])

            # check event number is different for each event
            first_evt_number = h5out.root.MC.extents[ 0][0]
            last_evt_number  = h5out.root.MC.extents[-1][0]
            assert first_evt_number != last_evt_number

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
    file_in     = os.path.join(ICDATADIR    , "Kr83_nexus_v5_03_00_ACTIVE_7bar_3evts*.MCRD.h5")
    file_out    = os.path.join(output_tmpdir, "Kr83_nexus_v5_03_00_ACTIVE_7bar_6evts.RWF.h5")
    second_file = os.path.join(ICDATADIR    , "Kr83_nexus_v5_03_00_ACTIVE_7bar_3evts_1.MCRD.h5")

    nevents_per_file = 3

    nrequired = 10
    conf = configure('dummy invisible_cities/config/diomira.conf'.split())
    conf.update(dict(run_number   = -4734,
                     files_in     = file_in,
                     file_out     = file_out,
                     event_range  = (0, nrequired)))
    conf["trigger_params"].update(dict(tr_channels = [18, 19]))

    diomira(**conf)

    with tb.open_file(file_out) as h5out:
        last_particle_list = h5out.root.MC.extents[:]['last_particle']
        last_hit_list      = h5out.root.MC.extents[:]['last_hit'     ]

        assert all(x<y for x, y in zip(last_particle_list, last_particle_list[1:]))
        assert all(x<y for x, y in zip(last_hit_list     , last_hit_list     [1:]))

        with tb.open_file(second_file) as h5second:
            nparticles_in_first_event = h5second.root.MC.extents[0]['last_particle'] + 1
            nhits_in_first_event      = h5second.root.MC.extents[0]['last_hit'     ] + 1

            assert last_particle_list[nevents_per_file] - last_particle_list[nevents_per_file - 1] == nparticles_in_first_event
            assert last_hit_list     [nevents_per_file] - last_hit_list     [nevents_per_file - 1] == nhits_in_first_event

def test_diomira_exact_result(ICDATADIR, output_tmpdir):
    file_in     = os.path.join(ICDATADIR    , "Kr83_nexus_v5_03_00_ACTIVE_7bar_3evts.MCRD.h5")
    file_out    = os.path.join(output_tmpdir,                       "exact_result_diomira.h5")
    true_output = os.path.join(ICDATADIR    ,  "Kr83_nexus_v5_03_00_ACTIVE_7bar_3evts.RWF.h5")

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


    tables = (     "MC/extents",  "MC/hits"   , "MC/particles", "MC/generators",
                   "RD/pmtrwf" ,  "RD/pmtblr" , "RD/sipmrwf"  ,
                  "Run/events" , "Run/runInfo",
              "Filters/trigger")
    with tb.open_file(true_output)  as true_output_file:
        with tb.open_file(file_out) as      output_file:
            for table in tables:
                got      = getattr(     output_file.root, table)
                expected = getattr(true_output_file.root, table)
                assert_tables_equality(got, expected)


def test_diomira_can_fix_random_seed(output_tmpdir):
    file_out    = os.path.join(output_tmpdir,                       "exact_result_diomira.h5")

    conf = configure("diomira invisible_cities/config/diomira.conf".split())
    conf.update(dict(random_seed = 123),
                file_out     = file_out)
    diomira(**conf)
