from math      import sqrt
from functools import partial

import numpy    as np
import networkx as nx

from itertools import combinations

from numpy.testing import assert_almost_equal

from pytest import fixture
from pytest import mark
from pytest import approx
from pytest import raises
parametrize = mark.parametrize

from hypothesis            import given
from hypothesis.strategies import lists
from hypothesis.strategies import floats
from hypothesis.strategies import builds
from hypothesis.strategies import integers

from networkx.generators.random_graphs import fast_gnp_random_graph

from .. evm.event_model  import BHit

from . paolina_functions import Voxel
from . paolina_functions import bounding_box
from . paolina_functions import find_extrema
from . paolina_functions import find_extrema_and_length
from . paolina_functions import blob_energies
from . paolina_functions import voxelize_hits
from . paolina_functions import shortest_paths
from . paolina_functions import make_track_graphs
from . paolina_functions import voxels_from_track_graph
from . paolina_functions import length
from . paolina_functions import Contiguity

from .. core.exceptions import NoHits
from .. core.exceptions import NoVoxels

def big_enough(hits):
    lo, hi = bounding_box(hits)
    bbsize = abs(hi-lo)
    return (bbsize > 0.1).all()

posn = floats(min_value=10, max_value=100)
ener = posn
bunch_of_hits = lists(builds(BHit, posn, posn, posn, ener),
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
    voxels = voxelize_hits(hits, voxel_dimensions, strict_voxel_size=False)

    if not hits:
        assert voxels == []

    def sum_energy(seq):
        return sum(e.E for e in seq)

    assert sum_energy(voxels) == approx(sum_energy(hits))


random_graph = builds(partial(fast_gnp_random_graph, p=0.5),
                      n=integers(min_value=0, max_value=10),
                      seed=integers())

@given(random_graph)
def test_voxels_from_track_return_node_voxels(graph):
    assert voxels_from_track_graph(graph) == graph.nodes()


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
def test_voxelize_hits_respects_voxel_dimensions(hits, requested_voxel_dimensions):
    voxels = voxelize_hits(hits, requested_voxel_dimensions, strict_voxel_size=True)
    unit   =                     requested_voxel_dimensions
    for v1, v2 in combinations(voxels, 2):
        distance_between_voxels = np.array(v2.XYZ) - np.array(v1.XYZ)
        off_by = distance_between_voxels % requested_voxel_dimensions
        assert (np.isclose(off_by, 0   ) |
                np.isclose(off_by, unit)).all()

@given(bunch_of_hits, box_sizes)
def test_voxelize_hits_gives_maximum_voxels_size(hits, requested_voxel_dimensions):
    voxels = voxelize_hits(hits, requested_voxel_dimensions, strict_voxel_size=False)
    for v in voxels:
        assert (v.size <= requested_voxel_dimensions).all()

@given(bunch_of_hits, box_sizes)
def test_voxelize_hits_strict_gives_required_voxels_size(hits, requested_voxel_dimensions):
    voxels = voxelize_hits(hits, requested_voxel_dimensions, strict_voxel_size=True)
    for v in voxels:
        assert v.size == approx(requested_voxel_dimensions)


@given(bunch_of_hits, box_sizes)
def test_make_voxel_graph_keeps_all_voxels(hits, voxel_dimensions):
    voxels = voxelize_hits    (hits  , voxel_dimensions)
    tracks = make_track_graphs(voxels)
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
    found = find_extrema_and_length(shortest_paths(weighted_graph))
    a, b = extrema
    assert a in found
    assert b in found


@given(builds(Voxel, posn, posn, posn, ener, box_sizes))
def test_find_extrema_single_voxel(voxel):
    g = nx.Graph()
    g.add_node(voxel)
    assert find_extrema(g) == (voxel, voxel)


def test_find_extrema_no_voxels():
    with raises(NoVoxels):
        find_extrema_and_length({})


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
    vox_size = np.array([1,1,1])
    voxels = [Voxel(x,y,z, E, vox_size) for (x,y,z,E) in voxel_spec]
    tracks  = make_track_graphs(voxels)

    assert len(tracks) == 1
    extrema = find_extrema(tracks[0])

    assert voxels[ 0] in extrema
    assert voxels[-1] in extrema

    return tracks[0], extrema


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


def test_voxelize_single_hit():
    hits = [BHit(1, 1, 1, 100)]
    vox_size = np.array([10,10,10], dtype=np.int16)
    assert len(voxelize_hits(hits, vox_size)) == 1


def test_length():
    voxel_spec = ((10,10,10,1),
                  (10,10,11,1),
                  (10,10,12,1),
                  (10,10,13,1),
                  (10,10,14,1),
                  (10,10,15,1),
                  (10,11,15,1),
                  (10,12,15,1),
                  (10,13,15,1),
                  (10,14,15,1),
                  (10,15,15,1)
    )
    voxels = [Voxel(x,y,z, E, np.array([1,1,1])) for (x,y,z,E) in voxel_spec]
    tracks  = make_track_graphs(voxels)

    assert len(tracks) == 1
    track_length = length(tracks[0])

    expected_length = 8 + np.sqrt(2)

    assert track_length == approx(expected_length)


@parametrize('contiguity, expected_length',
             ((Contiguity.FACE, 4),
              (Contiguity.CORNER, 2 * sqrt(2))))
def test_length_around_bend(contiguity, expected_length):
    # Make sure that we calculate the length along the track rather
    # that the shortcut
    voxel_spec = ((0,0,0),
                  (1,0,0),
                  (1,1,0),
                  (1,2,0),
                  (0,2,0))
    vox_size = np.array([1,1,1])
    voxels = [Voxel(x,y,z, 1, vox_size) for x,y,z in voxel_spec]
    tracks = make_track_graphs(voxels, contiguity=contiguity)
    assert len(tracks) == 1
    track_length = length(tracks[0])
    assert track_length == approx(expected_length)


@parametrize('contiguity, expected_length',
             (# Face contiguity requires 3 steps, each parallel to an axis
              (Contiguity.FACE,  1 + 1 + 1),
              # Edge continuity allows to cut one corner
              (Contiguity.EDGE,  1 + sqrt(2)),
              # Corner contiguity makes it possible to do in a single step
              (Contiguity.CORNER,    sqrt(3))))
def test_length_cuts_corners(contiguity, expected_length):
    "Make sure that we cut corners, if the contiguity allows"
    voxel_spec = ((0,0,0), # Extremum 1
                  (1,0,0),
                  (1,1,0),
                  (1,1,1)) # Extremum 2
    vox_size = np.array([1,1,1])
    voxels = [Voxel(x,y,z, 1, vox_size) for x,y,z in voxel_spec]
    tracks = make_track_graphs(voxels, contiguity=contiguity)

    assert len(tracks) == 1
    track_length = length(tracks[0])
    assert track_length == approx(expected_length)



FACE, EDGE, CORNER = Contiguity
@parametrize('contiguity,  proximity,          are_neighbours',
             ((FACE,      'share_face',            True),
              (FACE,      'share_edge',            False),
              (FACE,      'share_corner',          False),
              (FACE,      'share_nothing',         False),
              (FACE,      'share_nothing_algined', False),

              (EDGE,      'share_face',            True),
              (EDGE,      'share_edge',            True),
              (EDGE,      'share_corner',          False),
              (EDGE,      'share_nothing',         False),
              (EDGE,      'share_nothing_algined', False),

              (CORNER,    'share_face',            True),
              (CORNER,    'share_edge',            True),
              (CORNER,    'share_corner',          True),
              (CORNER,    'share_nothing',         False),
              (CORNER,    'share_nothing_algined', False),))
def test_contiguity(proximity, contiguity, are_neighbours):
    voxel_spec = dict(share_face            = ((0,0,0),
                                               (0,0,1)),
                      share_edge            = ((0,0,0),
                                               (0,1,1)),
                      share_corner          = ((0,0,0),
                                               (1,1,1)),
                      share_nothing         = ((0,0,0),
                                               (2,2,2)),
                      share_nothing_algined = ((0,0,0),
                                               (2,0,0)) )[proximity]
    expected_number_of_tracks = 1 if are_neighbours else 2
    voxels = [Voxel(x,y,z, 1, np.array([1,1,1])) for x,y,z in voxel_spec]
    tracks = make_track_graphs(voxels, contiguity=contiguity)

    assert len(tracks) == expected_number_of_tracks
    
