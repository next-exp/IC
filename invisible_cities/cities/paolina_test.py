import numpy    as np
import networkx as nx

from numpy.testing import assert_almost_equal

from pytest import fixture
from pytest import mark
from pytest import raises
parametrize = mark.parametrize

from hypothesis            import given
from hypothesis.strategies import lists
from hypothesis.strategies import floats
from hypothesis.strategies import builds
from hypothesis.strategies import integers

from . paolina import Hit
from . paolina import Voxel
from . paolina import bounding_box
from . paolina import find_extrema
from . paolina import blob_energies
from . paolina import voxelize_hits
from . paolina import shortest_paths
from . paolina import make_track_graphs

from .. core.exceptions import NoHits


def big_enough(hits):
    lo, hi = bounding_box(hits)
    bbsize = abs(hi-lo)
    return (bbsize > 0.1).all()

posn = floats(min_value=10, max_value=100)
ener = posn
bunch_of_hits = lists(builds(Hit, posn, posn, posn, ener),
                      min_size =  1,
                      max_size = 30).filter(big_enough)


box_dimension = floats(min_value = 1,
                       max_value = 5)
box_sizes = builds(np.array, lists(box_dimension,
                                   min_size = 3,
                                   max_size = 3))


def test_voxelize_hits_should_detect_no_hits():
    with raises(NoHits):
        voxelize_hits([], None)

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
                ( 1 , 5 , 1)], ( 2 , 4 )),
              ([( 1 , 2 , 1),
                ( 1 , 3 , 1),
                ( 1 , 4 , 2),
                ( 1 , 5 , 2)], ( 4 , 5 )),))
def test_find_extrema(spec, extrema):
    weighted_graph = nx.Graph([(a,b, dict(distance=d)) for (a,b,d) in spec])
    found = find_extrema(shortest_paths(weighted_graph))
    a, b = extrema
    assert a in found
    assert b in found

@given(builds(Voxel, posn, posn, posn, ener))
def test_find_extrema_single_voxel(voxel):
    g = nx.Graph()
    g.add_node(voxel)
    assert find_extrema(shortest_paths(g)) == (voxel, voxel)


@fixture(scope='module')
def track_extrema():
    voxel_spec = ((10,10,10,  1000),
                  (10,10,11,     1),
                  (10,10,12,     2),
                  (10,10,13,     4),
                  (10,10,14,     8),
                  (10,10,15,    16),
                  (10,11,15,    32),
                  (10,12,15,    64),
                  (10,13,15,   128),
                  (10,14,15,   265),
                  (10,15,15,   512),
                  (11,15,15,   256),
                  (12,15,15,   128),
                  (13,15,15,    64),
                  (14,15,15,    32),
                  (15,15,15,    16),
                  (16,16,16,     8),
                  (17,17,17,     4),
                  (18,18,18,     2),
                  (19,19,19,     1),
                  (20,20,20,  2000),
    )
    voxels = [Voxel(x,y,z, E) for (x,y,z,E) in voxel_spec]
    track, = make_track_graphs(voxels, np.array([1,1,1]))
    distances = shortest_paths(track)
    extrema = find_extrema(distances)

    assert voxels[ 0] in extrema
    assert voxels[-1] in extrema

    return track, extrema


@parametrize('radius, expected',
             ((0.5, (1000, 2000)), # Nothing but the endpoints
              (1.5, (1001, 2000)), # 10 10 10 is 1   away
              (1.9, (1001, 2001)), # 19 19 19 is 1.7 away
              (2.1, (1003, 2001)), # 10 10 12 is 2   away
              (3.1, (1007, 2001)), # 10 10 13 is 3   away
              (3.5, (1007, 2003)), # 18 18 18 is 3.4 away
              (4.5, (1015, 2003)), # 10 10 14 is 4   away
))
def test_blobs(track_extrema, radius, expected):
    track, extrema = track_extrema
    Ea, Eb = expected

    assert blob_energies(track, radius) == (Ea, Eb)
