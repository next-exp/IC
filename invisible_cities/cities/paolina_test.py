from . paolina import Hit
from . paolina import Voxel
from . paolina import voxelize_hits

def test_voxelize_hits_does_not_lose_energy():
    hits = [Hit(10.3, 20.9, 30.2, 123.2)]
    voxels = voxelize_hits(hits, None)

    def sum_energy(seq):
        return sum(e.E for e in seq)

    assert sum_energy(hits) == sum_energy(voxels)
