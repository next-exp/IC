import os
import tables as tb
import numpy  as np

from numpy.testing import assert_allclose

from .. evm.event_model import Voxel
from .. evm.event_model import Blob
from .. evm.event_model import Track
from .. evm.event_model import TrackCollection
from .  tracks_io       import tracks_writer

def test_tracks_writer(config_tmpdir, tracks_toy_data):

    tracks_filename, (event, time, track_no, track_len, voxel_no, X, Y, Z, E, size) = tracks_toy_data
    output_file = os.path.join(config_tmpdir, tracks_filename)

    with tb.open_file(output_file, 'w') as h5out:
        write = tracks_writer(h5out)
        tc = TrackCollection(0, 1.)
        voxels_trk = []
        voxels_blb1 = []
        voxels_blb2 = []
        for ii,(xv, yv, zv, ev, sv) in enumerate(zip(X, Y, Z, E, size)):
            vox = Voxel(xv, yv, zv, ev, sv)

            # For the moment, blob information is irrelevant.
            if(ii < 10):
                voxels_blb1.append(vox)
            elif(ii > 90):
                voxels_blb2.append(vox)

            voxels_trk.append(vox)

        blb1 = Blob(voxels_blb1[0],voxels_blb1)
        blb2 = Blob(voxels_blb2[0],voxels_blb2)
        trk = Track(voxels_trk,[blb1,blb2],track_len[0])
        tc.tracks.append(trk)

        write(tc)

    tdst = tb.open_file(output_file,'r')
    assert_allclose(event,    tdst.root.Voxels.Tracks[:]['event'])
    assert_allclose(time,     tdst.root.Voxels.Tracks[:]['time'])
    assert_allclose(track_no, tdst.root.Voxels.Tracks[:]['track_no'])
    assert_allclose(voxel_no, tdst.root.Voxels.Tracks[:]['voxel_no'])
    assert_allclose(X,        tdst.root.Voxels.Tracks[:]['X'])
    assert_allclose(Y,        tdst.root.Voxels.Tracks[:]['Y'])
    assert_allclose(Z,        tdst.root.Voxels.Tracks[:]['Z'])
    assert_allclose(E,        tdst.root.Voxels.Tracks[:]['E'])
    assert_allclose(size,     tdst.root.Voxels.Tracks[:]['size'])
