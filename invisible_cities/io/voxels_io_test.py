import os
import tables as tb
import numpy  as np

from numpy.testing import assert_allclose

from .. evm.event_model import Voxel
from .. evm.event_model import VoxelCollection
from .  voxels_io       import voxels_writer
from .  voxels_io       import load_voxels

def test_voxels_writer(config_tmpdir, voxels_toy_data):

    voxels_filename, (event, X, Y, Z, E, size) = voxels_toy_data
    output_file = os.path.join(config_tmpdir, 'toy_voxels.h5')

    with tb.open_file(output_file, 'w') as h5out:
        write = voxels_writer(h5out)
        voxels = VoxelCollection(0, 1.)
        for xv, yv, zv, ev, sv in zip(X, Y, Z, E, size):
            v = Voxel(xv, yv, zv, ev, sv)
            voxels.voxels.append(v)
        write(voxels)
    print(output_file)
    vdst = tb.open_file(output_file,'r')
    assert_allclose(event, vdst.root.Voxels.Voxels[:]['event'])
    assert_allclose(X,     vdst.root.Voxels.Voxels[:]['X'])
    assert_allclose(Y,     vdst.root.Voxels.Voxels[:]['Y'])
    assert_allclose(Z,     vdst.root.Voxels.Voxels[:]['Z'])
    assert_allclose(E,     vdst.root.Voxels.Voxels[:]['E'])
    assert_allclose(size,  vdst.root.Voxels.Voxels[:]['size'])

def test_load_voxels(config_tmpdir, voxels_toy_data):

    voxels_filename, (event, X, Y, Z, E, size) = voxels_toy_data

    voxels_dict = load_voxels(voxels_filename)
    vX =    [voxel.X    for voxel in voxels_dict[0].voxels]
    vY =    [voxel.Y    for voxel in voxels_dict[0].voxels]
    vZ =    [voxel.Z    for voxel in voxels_dict[0].voxels]
    vE =    [voxel.E    for voxel in voxels_dict[0].voxels]
    vsize = [voxel.size for voxel in voxels_dict[0].voxels]

    assert np.allclose(X, vX)
    assert np.allclose(Y, vY)
    assert np.allclose(Z, vZ)
    assert np.allclose(E, vE)
    assert np.allclose(size, vsize)
