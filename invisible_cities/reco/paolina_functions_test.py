import os

from math      import sqrt
from functools import partial

import tables as tb
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
from hypothesis            import settings
from hypothesis.strategies import lists
from hypothesis.strategies import floats
from hypothesis.strategies import integers
from hypothesis.strategies import builds
from hypothesis.strategies import integers

from networkx.generators.random_graphs import fast_gnp_random_graph

from .. evm.event_model import BHit
from .. evm.event_model import Voxel

from . paolina_functions import bounding_box
from . paolina_functions import energy_of_voxels_within_radius
from . paolina_functions import find_extrema
from . paolina_functions import find_extrema_and_length
from . paolina_functions import blob_energies
from . paolina_functions import blob_energies_and_hits
from . paolina_functions import blob_centre
from . paolina_functions import blob_centres
from . paolina_functions import hits_in_blob
from . paolina_functions import voxelize_hits
from . paolina_functions import shortest_paths
from . paolina_functions import make_track_graphs
from . paolina_functions import voxels_from_track_graph
from . paolina_functions import length
from . paolina_functions import Contiguity
from . paolina_functions import drop_end_point_voxels

from .. core.exceptions import NoHits
from .. core.exceptions import NoVoxels

from .. io.mcinfo_io    import load_mchits
from .. io.hits_io      import load_hits

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
radius = floats(min_value=1, max_value=100)

eps = 3e-12 # value that works for margin

fraction_zero_one = floats(min_value = 0,
                           max_value = 1)
min_n_of_voxels = integers(min_value = 3,
                           max_value = 10)


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


@settings(deadline=None)
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
        assert (v.size <= requested_voxel_dimensions + 4 * eps).all()


@given(bunch_of_hits, box_sizes)
def test_voxelize_hits_strict_gives_required_voxels_size(hits, requested_voxel_dimensions):
    voxels = voxelize_hits(hits, requested_voxel_dimensions, strict_voxel_size=True)
    for v in voxels:
        assert v.size == approx(requested_voxel_dimensions)


@given(bunch_of_hits, box_sizes)
def test_voxelize_hits_flexible_gives_correct_voxels_size(hits, requested_voxel_dimensions):
    voxels = voxelize_hits(hits, requested_voxel_dimensions, strict_voxel_size=False)
    positions = [v.pos for v in voxels]
    voxel_size = voxels[0].size

    def is_close_to_integer(n):
        return np.isclose(n, np.rint(n))

    for pos1, pos2 in combinations(positions, 2):
        separation_over_size = (pos2 - pos1) / voxel_size
        assert is_close_to_integer(separation_over_size).all()


@given(bunch_of_hits, box_sizes)
def test_hits_energy_in_voxel_is_equal_to_voxel_energy(hits, requested_voxel_dimensions):
    voxels = voxelize_hits(hits, requested_voxel_dimensions, strict_voxel_size=False)
    for v in voxels:
        assert sum(h.E for h in v.hits) == v.energy

def test_voxels_with_no_hits(ICDATADIR):
    hit_file = os.path.join(ICDATADIR, 'test_voxels_with_no_hits.h5')
    evt_number = 4803
    size = 15.
    vox_size = np.array([size,size,size],dtype=np.float16)

    with tb.open_file(hit_file, mode='r') as h5in:

        h5extents = h5in.root.MC.extents
        events_in_file = len(h5extents)

        for i in range(events_in_file):
            if h5extents[i]['evt_number'] == evt_number:
                evt_line = i
                break

        hits_dict = load_mchits(hit_file, (evt_line, evt_line+1))
        voxels = voxelize_hits(hits_dict[evt_number], vox_size, strict_voxel_size=False)
        for v in voxels:
            assert sum(h.E for h in v.hits) == v.energy


@given(bunch_of_hits, box_sizes)
def test_voxel_hits_are_same_as_original_ones(hits, requested_voxel_dimensions):
    voxels = voxelize_hits(hits, requested_voxel_dimensions, strict_voxel_size=False)
    hits_read_from_voxels = []
    for v in voxels:
        hits_read_from_voxels += v.hits

    hits_s_x   = sorted(hits,       key=lambda h: h.X)
    hits_s_xy  = sorted(hits_s_x,   key=lambda h: h.Y)
    hits_s_xyz = sorted(hits_s_xy,  key=lambda h: h.Z)
    hits_s_all = sorted(hits_s_xyz, key=lambda h: h.E)

    hits_read_from_voxels_s_x   = sorted(hits_read_from_voxels,       key=lambda h: h.X)
    hits_read_from_voxels_s_xy  = sorted(hits_read_from_voxels_s_x,   key=lambda h: h.Y)
    hits_read_from_voxels_s_xyz = sorted(hits_read_from_voxels_s_xy,  key=lambda h: h.Z)
    hits_read_from_voxels_s_all = sorted(hits_read_from_voxels_s_xyz, key=lambda h: h.E)

    assert  hits_read_from_voxels_s_all == hits_s_all


def test_hits_on_border_are_assigned_to_correct_voxel():
    z = 10.
    energy = 1.
    hits = [BHit( 5., 15., z, energy),
            BHit(15.,  5., z, energy),
            BHit(15., 15., z, energy),
            BHit(15., 25., z, energy),
            BHit(25., 15., z, energy)]

    vox_size = np.array([15,15,15], dtype=np.int16)
    voxels = voxelize_hits(hits, vox_size)

    assert len(voxels) == 3

    expected_hits = [[BHit( 5., 15., z, energy)],
                     [BHit(15.,  5., z, energy)],
                     [BHit(15., 15., z, energy),
                      BHit(15., 25., z, energy),
                      BHit(25., 15., z, energy)]]
    for v, hits in zip(voxels, expected_hits):
        assert v.hits == hits


@given(bunch_of_hits, box_sizes)
def test_make_voxel_graph_keeps_all_voxels(hits, voxel_dimensions):
    voxels = voxelize_hits    (hits  , voxel_dimensions)
    tracks = make_track_graphs(voxels)
    voxels_in_tracks = set().union(*(set(t.nodes()) for t in tracks))
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


def test_voxelize_single_hit():
    hits = [BHit(1, 1, 1, 100)]
    vox_size = np.array([10,10,10], dtype=np.int16)
    assert len(voxelize_hits(hits, vox_size)) == 1


@fixture(scope='module')
def voxels_without_hits():
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

    return voxels


def test_length():
    voxels = voxels_without_hits()
    tracks = make_track_graphs(voxels)

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


@given(bunch_of_hits, box_sizes, min_n_of_voxels, fraction_zero_one)
def test_energy_is_conserved_with_dropped_voxels(hits, requested_voxel_dimensions, min_voxels, fraction_zero_one):
    tot_initial_energy = sum(h.E for h in hits)
    voxels = voxelize_hits(hits, requested_voxel_dimensions, strict_voxel_size=False)
    ini_trks = make_track_graphs(voxels)
    ini_trk_energies = [sum(vox.E for vox in t.nodes()) for t in ini_trks]
    ini_trk_energies.sort()

    energies = [v.E for v in voxels]
    e_thr = min(energies) + fraction_zero_one * (max(energies) - min(energies))
    mod_voxels = drop_end_point_voxels(voxels, e_thr, min_voxels)
    tot_final_energy = sum(v.E for v in mod_voxels)
    final_trks = make_track_graphs(mod_voxels)
    final_trk_energies = [sum(vox.E for vox in t.nodes()) for t in final_trks]
    final_trk_energies.sort()

    assert tot_initial_energy == approx(tot_final_energy)
    assert np.allclose(ini_trk_energies, final_trk_energies)


def test_initial_voxels_are_the_same_after_dropping_voxels(ICDATADIR):

    # Get some test data: nothing interesting to see here
    hit_file = os.path.join(ICDATADIR, 'tracks_0000_6803_trigger2_v0.9.9_20190111_krth1600.h5')
    evt_number = 19
    e_thr = 5867.92
    min_voxels = 3
    size = 15.
    vox_size = np.array([size,size,size], dtype=np.float16)
    all_hits = load_hits(hit_file)
    hits = all_hits[evt_number].hits
    voxels = voxelize_hits(hits, vox_size, strict_voxel_size=False)

    # This is the core of the test: collect data before/after ...
    ante_energies  = [v.E   for v in voxels]
    ante_positions = [v.XYZ for v in voxels]
    mod_voxels = drop_end_point_voxels(voxels, e_thr, min_voxels)
    post_energies  = [v.E   for v in voxels]
    post_positions = [v.XYZ for v in voxels]

    ante_energies.sort()
    post_energies.sort()
    ante_positions.sort()
    post_positions.sort()

    # ... and make sure that nothing has changed
    assert len(ante_energies)  == len(post_energies)
    assert len(ante_positions) == len(post_positions)
    assert np.allclose(ante_energies,  post_energies)
    assert np.allclose(ante_positions, post_positions)


def test_tracks_with_dropped_voxels(ICDATADIR):
    hit_file = os.path.join(ICDATADIR, 'tracks_0000_6803_trigger2_v0.9.9_20190111_krth1600.h5')
    evt_number = 19
    e_thr = 5867.92
    min_voxels = 3
    size = 15.
    vox_size = np.array([size,size,size],dtype=np.float16)

    all_hits = load_hits(hit_file)
    hits = all_hits[evt_number].hits
    voxels = voxelize_hits(hits, vox_size, strict_voxel_size=False)
    ini_trks = make_track_graphs(voxels)
    initial_n_of_tracks = len(ini_trks)
    ini_energies = [sum(vox.E for vox in t.nodes()) for t in ini_trks]
    ini_n_voxels = np.array([len(t.nodes()) for t in ini_trks])

    mod_voxels = drop_end_point_voxels(voxels, e_thr, min_voxels)

    trks = make_track_graphs(mod_voxels)
    n_of_tracks = len(trks)
    energies = [sum(vox.E for vox in t.nodes()) for t in trks]
    n_voxels = np.array([len(t.nodes()) for t in trks])

    expected_diff_n_voxels = np.array([0, 0, 2])

    ini_energies.sort()
    energies.sort()

    assert initial_n_of_tracks == n_of_tracks
    assert np.allclose(ini_energies, energies)
    assert np.all(ini_n_voxels - n_voxels == expected_diff_n_voxels)


def test_voxel_drop_in_short_tracks():
    hits = [BHit(10, 10, 10, 1), BHit(26, 10, 10, 1)]
    voxels = voxelize_hits(hits, [15,15,15], strict_voxel_size=True)
    e_thr = sum(v.E for v in voxels) + 1.
    min_voxels = 0

    mod_voxels = drop_end_point_voxels(voxels, e_thr, min_voxels)

    assert len(mod_voxels) >= 1


@parametrize('radius, expected',
             ((10., ( 60,  20)),
              (12., ( 60,  60)),
              (14., (100,  60)),
              (16., (120,  80)),
              (18., (120,  80)),
              (20., (140,  80)),
              (22., (140, 100))
 ))
def test_blobs(radius, expected):
    hits = [BHit(105.0, 125.0, 77.7, 10),
            BHit( 95.0, 125.0, 77.7, 10),
            BHit( 95.0, 135.0, 77.7, 10),
            BHit(105.0, 135.0, 77.7, 10),
            BHit(105.0, 115.0, 77.7, 10),
            BHit( 95.0, 115.0, 77.7, 10),
            BHit( 95.0, 125.0, 79.5, 10),
            BHit(105.0, 125.0, 79.5, 10),
            BHit(105.0, 135.0, 79.5, 10),
            BHit( 95.0, 135.0, 79.5, 10),
            BHit( 95.0, 115.0, 79.5, 10),
            BHit(105.0, 115.0, 79.5, 10),
            BHit(115.0, 125.0, 79.5, 10),
            BHit(115.0, 125.0, 85.2, 10)]
    vox_size = np.array([15.,15.,15.],dtype=np.float16)
    voxels = voxelize_hits(hits, vox_size)
    tracks = make_track_graphs(voxels)

    assert len(tracks) == 1
    assert blob_energies(tracks[0], radius) == expected


@given(bunch_of_hits, box_sizes, radius)
def test_blob_hits_are_inside_radius(hits, voxel_dimensions, blob_radius):
    voxels = voxelize_hits(hits, voxel_dimensions)
    tracks = make_track_graphs(voxels)
    for t in tracks:
        Ea, Eb, hits_a, hits_b   = blob_energies_and_hits(t, blob_radius)
        centre_a, centre_b       = blob_centres(t, blob_radius)

        for h in hits_a:
            assert np.linalg.norm(h.XYZ - centre_a) < blob_radius
        for h in hits_b:
            assert np.linalg.norm(h.XYZ - centre_b) < blob_radius


@given(radius)
def test_paolina_functions_with_voxels_without_associated_hits(blob_radius):
    voxels = voxels_without_hits()
    tracks = make_track_graphs(voxels)
    for t in tracks:
        a, b = find_extrema(t)
        hits_a = hits_in_blob(t, blob_radius, a)
        hits_b = hits_in_blob(t, blob_radius, b)

        assert len(hits_a) == len(hits_b) == 0

        assert np.allclose(blob_centre(a), a.pos)
        assert np.allclose(blob_centre(b), b.pos)

        distances = shortest_paths(t)
        Ea = energy_of_voxels_within_radius(distances[a], blob_radius)
        Eb = energy_of_voxels_within_radius(distances[b], blob_radius)

        if Ea < Eb:
            assert np.allclose(blob_centres(t, blob_radius)[0], b.pos)
            assert np.allclose(blob_centres(t, blob_radius)[1], a.pos)
        else:
            assert np.allclose(blob_centres(t, blob_radius)[0], a.pos)
            assert np.allclose(blob_centres(t, blob_radius)[1], b.pos)

        assert blob_energies(t, blob_radius) != (0, 0)

        assert blob_energies_and_hits(t, blob_radius) != (0, 0, [], [])
