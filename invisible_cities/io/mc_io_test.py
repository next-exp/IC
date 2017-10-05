import tables as tb
import numpy as np
import os

from . mc_io import mc_track_writer

def test_non_consecutive_events(config_tmpdir, ICDIR):
    filein  = os.path.join(ICDIR, 'database/test_data/', 'tl_rwf_mctracks_test_3evts.h5')
    fileout = os.path.join(config_tmpdir, 'test_mctracks.h5')
    h5in    = tb.open_file(filein)
    h5out   = tb.open_file(fileout, 'w')

    mc_writer = mc_track_writer(h5out)
    mc_tracks = h5in.root.MC.MCTracks
    events_in = np.unique(h5in.root.MC.MCTracks[:]['event_indx'])

    #Skip one event (there are only 3 in the file)
    events_to_copy = events_in[::2]
    for evt in events_to_copy:
        mc_writer(mc_tracks, evt)

    events_out = np.unique(h5out.root.MC.MCTracks[:]['event_indx'])

    np.testing.assert_array_equal(events_to_copy, events_out)
