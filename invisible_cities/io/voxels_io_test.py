import os
import numpy  as np
from . mchits_io import load_mchits
from . mchits_io import load_mcparticles
from . mchits_io import load_mchits_nexus
from . mchits_io import load_mcparticles_nexus

def test_true_voxels_writer(config_tmpdir, voxels_toy_data):
    output_file = os.path.join(config_tmpdir, "test_hits.h5")

    _, (event, X, Y, Z, E, size) = voxels_toy_data

    with tb.open_file(output_file, 'w') as h5out:
        write = true_voxels_writer(h5out)
        voxels = VoxelCollection()
        for xv, yv, zv, ev in zip(X, Y, Z, E):
            v = Voxel(xv, yv, zv, ev)
            voxels.voxels.append(v)
        write(voxels.voxels)

    dst = load_dst(output_file, group = "RECO", node = "Events")
    assert_allclose(event, dst.event.values)
    assert_allclose(X,    dst.X.values)
    assert_allclose(Y,    dst.Y.values)
    assert_allclose(Z,    dst.Z.values)
    assert_allclose(E,    dst.E.values)
    assert_allclose(size, dst.size.values)
