from hypothesis            import given
from hypothesis.strategies import lists
from hypothesis.strategies import floats
from hypothesis.strategies import builds

from . paolina import Hit
from . paolina import Voxel
from . paolina import voxelize_hits

posn = floats(min_value=0, max_value=100)
ener = posn

@given(lists(builds(Hit, posn, posn, posn, ener),
             min_size = 0,
             max_size = 30))
def test_voxelize_hits_does_not_lose_energy(hits):
    voxels = voxelize_hits(hits, None)

    def sum_energy(seq):
        return sum(e.E for e in seq)

    assert sum_energy(hits) == sum_energy(voxels)
