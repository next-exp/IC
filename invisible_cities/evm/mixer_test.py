import os
import glob
import pandas as pd
import tables as tb

from ..io.dst_io import load_dst, load_dsts

from ..database.load_db import RadioactivityData

from .mixer import get_file_number
from .mixer import Event_Mixer
from .mixer import get_mixer_nevents


def test_Event_Mixer_writes_all_tables(ICDATADIR, output_tmpdir):
    '''
    Runs the event mixer class to test that the output files contain all the expected tables
    '''

    # mixer config
    inpath   = os.path.join(ICDATADIR, "mixer/{g4volume}/{isotope}/*_test.h5")
    outpath  = os.path.join(output_tmpdir, "mixer_{file_number}_test_all_tables.h5")
    nevents_per_file = 10

    # create event dataframe
    df = pd.DataFrame(columns = ["G4Volume", "Isotope", "nevts"])
    df.loc[len(df)] = ["ACTIVE", "0nubb", 3]
    df.loc[len(df)] = ["ACTIVE", "2nubb", 10]
    df.nevts = df.nevts.astype(int)

    # run mixer
    mixer = Event_Mixer(inpath, outpath, df, nevents_per_file)
    mixer.run()

    filenames = sorted(glob.glob(outpath.format(file_number="*")), key=get_file_number)
    for filename in filenames:
        with tb.open_file(filename, "r") as h5out:
            assert hasattr(h5out.root, "Run/events")
            assert hasattr(h5out.root, "DST/Events")
            assert hasattr(h5out.root, "Tracking/Tracks")
            assert hasattr(h5out.root, "Summary/Events")
            assert hasattr(h5out.root, "MC/hits")
            assert hasattr(h5out.root, "MC/particles")
            assert hasattr(h5out.root, "MC/sns_response")


def test_Event_Mixer_nevents(ICDATADIR, output_tmpdir):
    '''
    Runs the event mixer class to test that the output files contain
    the total number of events per file and the number of events of each component
    provided as input. Also checks the unique event-id
    '''

    # mixer config
    inpath   = os.path.join(ICDATADIR, "mixer/{g4volume}/{isotope}/*_test.h5")
    outpath  = os.path.join(output_tmpdir, "mixer_{file_number}_test_nevents.h5")
    nevents_per_file = 10

    # create event dataframe
    df = pd.DataFrame(columns = ["G4Volume", "Isotope", "nevts"])
    df.loc[len(df)] = ["ACTIVE", "0nubb", 3]
    df.loc[len(df)] = ["ACTIVE", "2nubb", 10]
    df.nevts = df.nevts.astype(int)

    # run mixer
    mixer = Event_Mixer(inpath, outpath, df, nevents_per_file)
    mixer.run()

    # test total events per file
    filenames = sorted(glob.glob(outpath.format(file_number="*")), key=get_file_number)
    nevents_per_file = min(nevents_per_file, df.nevts.sum())
    for filename in filenames[:-1]: # last file could not be full
        events = load_dst(filename, "Run", "events")
        assert (len(events) == nevents_per_file)

    # test total events per component
    indexes = ["G4Volume", "Isotope", "evt_number", "file"]
    nevents_df = mixer.nevents_df.reset_index().set_index(indexes[:2])

    for key in ("events", "dst", "tracking", "summary"):
        dst = load_dsts(filenames, *mixer.tables[key])
        nev = dst.groupby(indexes[:2]) \
                 .apply(lambda df: df.set_index(indexes[2:]).index.nunique()).to_frame()\
                 .rename({0:"nevts"}, axis=1)

        pd.testing.assert_frame_equal(nev, nevents_df)

    # test unique-id
    for key in ("events", "dst", "tracking", "summary"):
        dst = load_dsts(filenames, *mixer.tables[key])
        assert dst.event.nunique() == nevents_df.nevts.sum()


def test_get_mixer_nevents():
    '''
    Tests that get_mixer_nevents returns the expected number of events for
    a simple user case
    '''

    detector_db = "next100"
    isotopes = ["Bi214", "Co60"]
    exposure = 1 # dummy

    got = get_mixer_nevents(exposure, detector_db, isotopes)
    got = got.set_index(["G4Volume", "Isotope"]).nevts

    act, eff = RadioactivityData(detector_db)
    act = act[act.Isotope.isin(isotopes)].set_index(["G4Volume", "Isotope"])
    eff = eff[eff.Isotope.isin(isotopes)].set_index(["G4Volume", "Isotope"])
    expected = (act.TotalActivity * eff.MCEfficiency * exposure).dropna().rename("nevts")

    pd.testing.assert_series_equal(got, expected)
