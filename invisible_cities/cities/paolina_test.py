import numpy    as np
import networkx as nx

from numpy.testing import assert_almost_equal

from pytest import mark
parametrize = mark.parametrize

from hypothesis            import given
from hypothesis.strategies import lists
from hypothesis.strategies import floats
from hypothesis.strategies import builds

from . paolina import Hit
from . paolina import Voxel
from . paolina import bounding_box
from . paolina import find_extrema
from . paolina import voxelize_hits
from . paolina import make_track_graphs


def big_enough(hits):
    lo, hi = bounding_box(hits)
    bbsize = abs(hi-lo)
    return (bbsize > 0.1).all()

posn = floats(min_value=10, max_value=100)
ener = posn
bunch_of_hits = lists(builds(Hit, posn, posn, posn, ener),
                      min_size =  0,
                      max_size = 30).filter(big_enough)


box_dimension = floats(min_value = 1,
                       max_value = 5)
box_sizes = builds(np.array, lists(box_dimension,
                                   min_size = 3,
                                   max_size = 3))


@given(bunch_of_hits)
def test_bounding_box(hits):
    if not len(hits): # TODO: deal with empty sequences
        return

    lo, hi = bounding_box(hits)

    mins = [float(' inf')] * 3
    maxs = [float('-inf')] * 3

    for hit in hits:
        assert lo[0] <= hit.pos[0] <= hi[0]
        assert lo[1] <= hit.pos[1] <= hi[1]
        assert lo[2] <= hit.pos[2] <= hi[2]

        for d in range(3):
            mins[d] = min(mins[d], hit.pos[d])
            maxs[d] = max(maxs[d], hit.pos[d])

    for d in range(3):
        assert_almost_equal(mins[d], lo[d])
        assert_almost_equal(maxs[d], hi[d])


@given(bunch_of_hits, box_sizes)
def test_voxelize_hits_does_not_lose_energy(hits, voxel_dimensions):
    voxels = voxelize_hits(hits, voxel_dimensions)

    if not hits:
        assert voxels == []

    def sum_energy(seq):
        return sum(e.E for e in seq)

    assert_almost_equal(sum_energy(hits), sum_energy(voxels))


@given(bunch_of_hits, box_sizes)
def test_voxelize_hits_keeps_bounding_box(hits, voxel_dimensions):
    voxels = voxelize_hits(hits, voxel_dimensions)

    hlo, hhi = bounding_box(hits)
    vlo, vhi = bounding_box(voxels)

    vlo -= 0.5 * voxel_dimensions
    vhi += 0.5 * voxel_dimensions

    assert (vlo <= hlo).all()
    assert (vhi >= hhi).all()

@given(bunch_of_hits, box_sizes)
def test_make_voxel_graph_keeps_all_voxels(hits, voxel_dimensions):
    voxels = voxelize_hits    (hits  , voxel_dimensions)
    tracks = make_track_graphs(voxels, voxel_dimensions)
    voxels_in_tracks = set().union(*(set(t.nodes_iter()) for t in tracks))
    assert set(voxels) == voxels_in_tracks


@parametrize(' spec,           extrema',
             (([( 1 , 2 , 3)], ( 1 , 2 )),
              ([('a','b', 4)], ('a','b')),
              ([( 1 , 2 , 3),
                ( 2 , 3 , 1)], ( 1 , 3 )),
              ([( 1 , 2 , 3),
                ( 1 , 3 , 1),
                ( 1 , 4 , 2),
                ( 1 , 5 , 1)], ( 2 , 4 )),))
def test_find_extrema(spec, extrema):
    weighted_graph = nx.Graph([(a,b, dict(distance=d)) for (a,b,d) in spec])
    assert find_extrema(weighted_graph) == set(extrema)
