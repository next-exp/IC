import os
import tables as tb

from numpy.testing import assert_allclose

from .. evm.event_model import Voxel
from .. evm.event_model import VoxelCollection
from .  dst_io          import load_dst
from .  voxels_io       import true_voxels_writer

def test_true_voxels_writer(config_tmpdir, voxels_toy_data):
    output_file = os.path.join(config_tmpdir, "test_voxels.h5")

    _, (event, X, Y, Z, E, size) = voxels_toy_data

    with tb.open_file(output_file, 'w') as h5out:
        write = true_voxels_writer(h5out)
        voxels = VoxelCollection([])
        for xv, yv, zv, ev, sv in zip(X, Y, Z, E, size):
            v = Voxel(xv, yv, zv, ev, sv)
            voxels.voxels.append(v)
        write(event[0],voxels.voxels)

    vdst = tb.open_file(output_file,'r')
    assert_allclose(event, vdst.root.TrueVoxels.Voxels[:]['event'])
    assert_allclose(X,     vdst.root.TrueVoxels.Voxels[:]['X'])
    assert_allclose(Y,     vdst.root.TrueVoxels.Voxels[:]['Y'])
    assert_allclose(Z,     vdst.root.TrueVoxels.Voxels[:]['Z'])
    assert_allclose(E,     vdst.root.TrueVoxels.Voxels[:]['E'])
    assert_allclose(size,  vdst.root.TrueVoxels.Voxels[:]['size'])

def test_load_voxels():
