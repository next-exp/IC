import os

from math      import sqrt
from functools import partial

import numpy    as np
import pandas   as pd
import networkx as nx

from itertools import combinations
from operator  import attrgetter

from numpy.testing import assert_almost_equal

from pytest import fixture
from pytest import mark
from pytest import approx
from pytest import raises
parametrize = mark.parametrize

from hypothesis            import given
from hypothesis            import settings
from hypothesis            import assume
from hypothesis.strategies import composite
from hypothesis.strategies import lists
from hypothesis.strategies import floats
from hypothesis.strategies import integers
from hypothesis.strategies import builds

from networkx.generators.random_graphs import fast_gnp_random_graph

from .. evm.event_model import BHit
from .. evm.event_model import Hit
from .. evm.event_model import Cluster
from .. evm.event_model import Voxel

from . paolina_functions import bounding_box
from . paolina_functions import round_hits_positions_in_place
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
from . paolina_functions import drop_end_point_voxels
from . paolina_functions import make_tracks
from . paolina_functions import get_track_energy

from .. core                import system_of_units as units
from .. core.core_functions import in_range
from .. core.exceptions     import NoHits
from .. core.exceptions     import NoVoxels
from .. core.testing_utils  import assert_dataframes_close
from .. core.testing_utils  import an_instance_of

from .. io.mcinfo_io    import cast_mchits_to_dict
from .. io.mcinfo_io    import load_mchits_df
from .. io.hits_io      import load_hits

from .. types.ic_types import xy
from .. types.symbols  import Contiguity
from .. types.symbols  import HitEnergy


def big_enough(hits):
    lo, hi = bounding_box(hits)
    bbsize = abs(hi-lo)
    return (bbsize > 0.1).all()

@composite
def single_hits(draw):
    return pd.DataFrame(dict( event    = 0
                            , time     = 0
                            , npeak    = 0
                            , x_peak   = draw(floats  (-10,  10))
                            , y_peak   = draw(floats  (-10,  10))
                            , X        = draw(floats  (-10,  10))
                            , Y        = draw(floats  (-10,  10))
                            , Z        = draw(floats  ( 50, 100))
                            , Q        = draw(floats  (  1, 100))
                            , E        = draw(floats  ( 50, 100))
                            , Ec       = draw(floats  ( 50, 100))
                            , track_id = draw(integers(  0,  10))
                            , Ep       = draw(floats  ( 50, 100))
                            ), index=[0])

@composite
def bunch_of_hits(draw):
    strat = lists(single_hits(), min_size=1, max_size=30)
    hits  = draw(strat)
    hits  = pd.concat(hits, ignore_index=True)
    assume(big_enough(hits))
    return hits

@composite
def single_voxels(draw):
    return Voxel( x = draw(integers(-5, 5)) * 15.55
                , y = draw(integers(-5, 5)) * 15.55
                , z = draw(integers(-5, 5)) *  4.00
                , E = draw(floats(1.0, 234.0))
                , size = [15.55, 15.55, 4.00]
                , hits = []
                )

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


@an_instance_of(single_hits())
def test_voxelize_hits_should_detect_no_hits_demoo(hits):
    empty = hits.iloc[:0]
    with raises(NoHits):
        voxelize_hits(empty, None)


@given(bunch_of_hits())
def test_bounding_box(hits):
    lo, hi = bounding_box(hits)

    assert np.all(in_range(hits.X, lo[0], hi[0], right_closed=True))
    assert np.all(in_range(hits.Y, lo[1], hi[1], right_closed=True))
    assert np.all(in_range(hits.Z, lo[2], hi[2], right_closed=True))

    for col, l, h in zip("X Y Z".split(), lo, hi):
        assert_almost_equal(hits[col].min(), l)
        assert_almost_equal(hits[col].max(), h)


@given(bunch_of_hits(), box_sizes)
def test_voxelize_hits_does_not_lose_energy(hits, voxel_dimensions):
    voxels = voxelize_hits(hits, voxel_dimensions, strict_voxel_size=False)
    total_voxel_e = sum(v.E for v in voxels)
    total_hits_e  = sum(hits.E)

    assert total_voxel_e == approx(total_hits_e)


@given(bunch_of_hits())
def test_round_hits_positions_in_place(hits):
    """
    Override xyz such that all values fall below the rounding decimal place. We
    also multiply some values by -1 to include negative numbers. The maximum
    absolute value xyz can have is 100. After multiplying by 1e-7, the maximum
    absolute value is 1e-5. After rounding, the only possible values are 0, 1e-5
    or -1e-5.
    """
    hits.loc[:, "X Y Z".split()] = hits["X Y Z".split()].values * 0.999e-7 * [-1, 1, -1]

    round_hits_positions_in_place(hits)

    assert np.all(np.in1d(hits["X Y Z".split()], [0, 1e-5, -1e-5]))


@an_instance_of(single_hits())
def test_round_hits_positions_in_place_empty_input(hits):
    """
    It simply should not crash.
    """
    hits = hits.iloc[:0]
    round_hits_positions_in_place(hits)
    assert len(hits) == 0


@an_instance_of(single_hits())
def test_round_hits_positions_in_place_non_finite_values(hit):
    """
    Override xyz with np.nan and np.inf, ensure values are not changed.
    """
    hit.loc[:, "X"] =  np.nan
    hit.loc[:, "Y"] =  np.inf
    hit.loc[:, "Z"] = -np.inf

    round_hits_positions_in_place(hit)
    assert np.all(np.isclose(hit["X Y Z".split()].values[0], np.array([np.nan, np.inf, -np.inf]), equal_nan=True))



random_graph = builds(partial(fast_gnp_random_graph, p=0.5),
                      n=integers(min_value=0, max_value=10),
                      seed=integers())

@given(random_graph)
def test_voxels_from_track_return_node_voxels(graph):
    assert voxels_from_track_graph(graph) == graph.nodes()


@mark.skip(reason="bounding box no longer works on voxels")
@given(bunch_of_hits(), box_sizes)
def test_voxelize_hits_keeps_bounding_box(hits, voxel_dimensions):
    voxels = voxelize_hits(hits, voxel_dimensions)

    hlo, hhi = bounding_box(hits)
    vlo, vhi = bounding_box(voxels)

    vlo -= 0.5 * voxel_dimensions
    vhi += 0.5 * voxel_dimensions

    assert (vlo <= hlo).all()
    assert (vhi >= hhi).all()


@settings(deadline=None)
@given(bunch_of_hits(), box_sizes)
def test_voxelize_hits_respects_voxel_dimensions(hits, requested_voxel_dimensions):
    voxels = voxelize_hits(hits, requested_voxel_dimensions, strict_voxel_size=True)
    unit   =                     requested_voxel_dimensions
    for v1, v2 in combinations(voxels, 2):
        distance_between_voxels = np.array(v2.XYZ) - np.array(v1.XYZ)
        off_by = distance_between_voxels % requested_voxel_dimensions
        assert (np.isclose(off_by, 0   ) |
                np.isclose(off_by, unit)).all()


@given(bunch_of_hits(), box_sizes)
def test_voxelize_hits_gives_maximum_voxels_size(hits, requested_voxel_dimensions):
    voxels = voxelize_hits(hits, requested_voxel_dimensions, strict_voxel_size=False)
    for v in voxels:
        assert (v.size <= requested_voxel_dimensions + 4 * eps).all()


@given(bunch_of_hits(), box_sizes)
def test_voxelize_hits_strict_gives_required_voxels_size(hits, requested_voxel_dimensions):
    voxels = voxelize_hits(hits, requested_voxel_dimensions, strict_voxel_size=True)
    for v in voxels:
        assert v.size == approx(requested_voxel_dimensions)


@given(bunch_of_hits(), box_sizes)
def test_voxelize_hits_flexible_gives_correct_voxels_size(hits, requested_voxel_dimensions):
    voxels = voxelize_hits(hits, requested_voxel_dimensions, strict_voxel_size=False)
    positions = [v.pos for v in voxels]
    voxel_size = voxels[0].size

    def is_close_to_integer(n):
        return np.isclose(n, np.rint(n))

    for pos1, pos2 in combinations(positions, 2):
        separation_over_size = (pos2 - pos1) / voxel_size
        assert is_close_to_integer(separation_over_size).all()


@given(bunch_of_hits(), box_sizes)
def test_hits_energy_in_voxel_is_equal_to_voxel_energy(hits, requested_voxel_dimensions):
    voxels = voxelize_hits(hits, requested_voxel_dimensions, strict_voxel_size=False)
    for v in voxels:
        assert v.hits.E.sum() == approx(v.E)

def test_voxels_with_no_hits(ICDATADIR):
    hit_file = os.path.join(ICDATADIR, 'test_voxels_with_no_hits.h5')
    evt_number = 4803
    size = 15.
    vox_size = np.array([size,size,size],dtype=np.float16)

    hits_df = load_mchits_df(hit_file)
    hits_df = hits_df.loc[evt_number].rename(columns=dict(x="X", y="Y", z="Z", energy="E"))
    voxels  = voxelize_hits(hits_df, vox_size, strict_voxel_size=True)

    for v in voxels:
        assert v.hits.E.sum() == approx(v.E)


@given(bunch_of_hits(), box_sizes)
def test_voxel_hits_are_same_as_original_ones(hits, requested_voxel_dimensions):
    voxels = voxelize_hits(hits, requested_voxel_dimensions, strict_voxel_size=False)
    hits_from_voxels = []
    for v in voxels:
        hits_from_voxels.append(v.hits)

    hits_from_voxels = pd.concat(hits_from_voxels, ignore_index=True)

    # need to sort and reset index for a reliable comparison
    hits             = hits            .sort_values("X Y Z E".split()).reset_index(drop=True)
    hits_from_voxels = hits_from_voxels.sort_values("X Y Z E".split()).reset_index(drop=True)

    assert_dataframes_close(hits, hits_from_voxels)


def test_hits_on_border_are_assigned_to_correct_voxel():
    z = 10.
    energy = 1.
    hits = pd.DataFrame(dict( X = [ 5, 15, 15, 15, 25]
                            , Y = [15,  5, 15, 25, 15]
                            , Z = z
                            , E = energy
                            ), dtype=float)

    vox_size = np.array([15,15,15], dtype=np.int16)
    voxels = voxelize_hits(hits, vox_size)

    assert len(voxels) == 3

    expected_hits = [ hits.iloc[0:1], hits.iloc[1:2], hits.loc[2:] ]
    for v, exp_hits in zip(voxels, expected_hits):
        assert_dataframes_close(v.hits, exp_hits)


@given(bunch_of_hits(), box_sizes)
def test_make_voxel_graph_keeps_all_voxels(hits, voxel_dimensions):
    voxels = voxelize_hits    (hits  , voxel_dimensions)
    tracks = make_track_graphs(voxels)
    voxels_in_tracks = set().union(*(set(t.nodes()) for t in tracks))
    assert set(voxels) == voxels_in_tracks


@given(single_voxels())
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


@an_instance_of(single_hits())
def test_voxelize_single_hit(hit):
    vox_size = np.array([10,10,10], dtype=np.int16)
    assert len(voxelize_hits(hit, vox_size)) == 1


@fixture(scope='module')
def voxels_without_hits():
    voxel_spec = ((10,10,10,1),
                  (10,10,11,2),
                  (10,10,12,2),
                  (10,10,13,2),
                  (10,10,14,2),
                  (10,10,15,2),
                  (10,11,15,2),
                  (10,12,15,2),
                  (10,13,15,2),
                  (10,14,15,2),
                  (10,15,15,2))

    hits   = pd.DataFrame(columns="X Y Z E Ep".split())
    voxels = [Voxel(x,y,z, E, np.array([1,1,1]), hits=hits) for (x,y,z,E) in voxel_spec]

    return voxels


def test_length(voxels_without_hits):
    voxels = voxels_without_hits
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


@given(bunch_of_hits(), box_sizes, min_n_of_voxels, fraction_zero_one)
def test_energy_is_conserved_with_dropped_voxels(hits, requested_voxel_dimensions, min_voxels, fraction_zero_one):
    tot_initial_energy = hits.E.sum()
    voxels = voxelize_hits(hits, requested_voxel_dimensions, strict_voxel_size=False)
    ini_trks = make_track_graphs(voxels)
    ini_trk_energies = [sum(vox.E for vox in t.nodes()) for t in ini_trks]
    ini_trk_energies.sort()

    energies = [v.E for v in voxels]
    e_thr = min(energies) + fraction_zero_one * (max(energies) - min(energies))
    mod_voxels, _ = drop_end_point_voxels(voxels, e_thr, min_voxels)
    tot_final_energy = sum(v.E for v in mod_voxels)
    final_trks = make_track_graphs(mod_voxels)
    final_trk_energies = [sum(vox.E for vox in t.nodes()) for t in final_trks]
    final_trk_energies.sort()

    assert tot_initial_energy == approx(tot_final_energy)
    assert np.allclose(ini_trk_energies, final_trk_energies)


@mark.parametrize("energy_type", HitEnergy)
@given(hits                       = bunch_of_hits(),
       requested_voxel_dimensions = box_sizes,
       min_voxels                 = min_n_of_voxels,
       fraction_zero_one          = fraction_zero_one)
def test_dropped_voxels_have_nan_energy(hits, requested_voxel_dimensions, min_voxels, fraction_zero_one, energy_type):
    voxels            = voxelize_hits(hits, requested_voxel_dimensions, strict_voxel_size=False, energy_type=energy_type)
    energies          = [v.E for v in voxels]
    e_thr             = min(energies) + fraction_zero_one * (max(energies) - min(energies))
    _, dropped_voxels = drop_end_point_voxels(voxels, e_thr, min_voxels)
    for voxel in dropped_voxels:
        assert np.isnan(voxel.E)
        assert voxel.hits[energy_type.value].isna().all()


@mark.parametrize("energy_type", HitEnergy)
@given(hits                       = bunch_of_hits(),
       requested_voxel_dimensions = box_sizes,
       min_voxels                 = min_n_of_voxels,
       fraction_zero_one          = fraction_zero_one)
def test_drop_end_point_voxels_doesnt_modify_other_energy_types(hits, requested_voxel_dimensions, min_voxels, fraction_zero_one, energy_type):
    voxels     = voxelize_hits(hits, requested_voxel_dimensions, strict_voxel_size=False, energy_type=energy_type)
    voxels     = sorted(voxels, key=attrgetter("xyz"))
    energies   = [v.E for v in voxels]
    e_thr      = min(energies) + fraction_zero_one * (max(energies) - min(energies))
    mod, drop  = drop_end_point_voxels(voxels, e_thr, min_voxels)
    new_voxels = sorted(mod + drop, key=attrgetter("xyz"))

    for e_type in HitEnergy:
        if e_type is energy_type: continue

        for v_before, v_after in zip(voxels, new_voxels):
            e_before = v_before.hits[e_type.value]
            e_after  = v_after .hits[e_type.value]
            assert np.allclose(e_before, e_after)


@mark.parametrize("energy_type", HitEnergy)
@given(hits                       = bunch_of_hits(),
       requested_voxel_dimensions = box_sizes,
       min_voxels                 = min_n_of_voxels,
       fraction_zero_one          = fraction_zero_one)
def test_drop_voxels_voxel_energy_is_sum_of_hits_general(hits, requested_voxel_dimensions, min_voxels, fraction_zero_one, energy_type):
    voxels        = voxelize_hits(hits, requested_voxel_dimensions, strict_voxel_size=False, energy_type=energy_type)
    energies      = [v.E for v in voxels]
    e_thr         = min(energies) + fraction_zero_one * (max(energies) - min(energies))
    mod_voxels, _ = drop_end_point_voxels(voxels, e_thr, min_voxels)
    for v in mod_voxels:
        assert np.isclose(v.E, v.hits[energy_type.value].sum())


@mark.parametrize("energy_type", HitEnergy)
@given(hits                       = bunch_of_hits(),
       requested_voxel_dimensions = box_sizes,
       min_voxels                 = min_n_of_voxels,
       fraction_zero_one          = fraction_zero_one)
def test_drop_end_point_voxels_constant_number_of_voxels_and_hits(hits, requested_voxel_dimensions, min_voxels, fraction_zero_one, energy_type):
    voxels           = voxelize_hits(hits, requested_voxel_dimensions, strict_voxel_size=False, energy_type=energy_type)
    energies         = [v.E for v in voxels]
    e_thr            = min(energies) + fraction_zero_one * (max(energies) - min(energies))
    new_voxels       = drop_end_point_voxels(voxels, e_thr, min_voxels)
    (mod_voxels,
     dropped_voxels) = new_voxels
    assert len(mod_voxels) + len(dropped_voxels) == len(voxels)
    assert sum(len(v.hits) for v in mod_voxels + dropped_voxels) == len(hits)


def test_initial_voxels_are_the_same_after_dropping_voxels(ICDATADIR):

    # Get some test data: nothing interesting to see here
    hit_file = os.path.join(ICDATADIR, 'tracks_0000_6803_trigger2_v0.9.9_20190111_krth1600.h5')
    evt_number = 19
    e_thr = 5867.92
    min_voxels = 3
    size = 15.
    vox_size = np.array([size,size,size], dtype=np.float16)
    hits = pd.read_hdf(hit_file, "/RECO/Events").set_index("event").loc[evt_number]
    voxels = voxelize_hits(hits, vox_size, strict_voxel_size=False)

    # This is the core of the test: collect data before/after ...
    ante_energies  = [v.E   for v in voxels]
    ante_positions = [v.XYZ for v in voxels]
    mod_voxels, _ = drop_end_point_voxels(voxels, e_thr, min_voxels)
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

    hits = pd.read_hdf(hit_file, "/RECO/Events").set_index("event").loc[evt_number]
    voxels = voxelize_hits(hits, vox_size, strict_voxel_size=False)
    ini_trks = make_track_graphs(voxels)
    initial_n_of_tracks = len(ini_trks)
    ini_energies = [sum(vox.E for vox in t.nodes()) for t in ini_trks]
    ini_n_voxels = np.array([len(t.nodes()) for t in ini_trks])

    mod_voxels, _ = drop_end_point_voxels(voxels, e_thr, min_voxels)

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


def test_drop_voxels_deterministic(ICDATADIR):
    hit_file   = os.path.join(ICDATADIR, 'tracks_0000_6803_trigger2_v0.9.9_20190111_krth1600.h5')
    evt_number = 19
    e_thr      = 5867.92
    min_voxels = 3
    vox_size   = [15.] * 3

    hits            = pd.read_hdf(hit_file, "/RECO/Events").set_index("event").loc[evt_number]
    voxels          = voxelize_hits(hits, vox_size, strict_voxel_size=False)
    mod_voxels  , _ = drop_end_point_voxels(sorted(voxels, key = lambda v:v.E, reverse = False), e_thr, min_voxels)
    mod_voxels_r, _ = drop_end_point_voxels(sorted(voxels, key = lambda v:v.E, reverse = True ), e_thr, min_voxels)

    for v1, v2 in zip(sorted(mod_voxels, key = lambda v:v.E), sorted(mod_voxels_r, key = lambda v:v.E)):
        assert np.isclose(v1.E, v2.E)


def test_voxel_drop_in_short_tracks():
    hits = pd.DataFrame(dict(X=[10, 26], Y=[10,10], Z=[10,10], E=[1,1]))
    voxels = voxelize_hits(hits, [15,15,15], strict_voxel_size=True)
    e_thr = sum(v.E for v in voxels) + 1.
    min_voxels = 0

    mod_voxels, _ = drop_end_point_voxels(voxels, e_thr, min_voxels)

    assert len(mod_voxels) >= 1


def test_drop_voxels_voxel_energy_is_sum_of_hits():
    def make_hit(x, y, z, e):
        return pd.DataFrame(dict(X=x, Y=y, Z=z, E=e, Ep=e), index=[0])

    def make_voxel(x, y, z, e, hits):
        return Voxel(x, y, z, e, size=5, e_type=HitEnergy.Ep, hits=pd.concat(hits, ignore_index=True))
    # Create a track with an extreme to be dropped and two hits at the same
    # distance from the barycenter of the the voxel to be dropped with
    # different energies in the hits *and* in the voxels
    voxels = [make_voxel( 0,  0, 0, 0.1, [make_hit( 0,  0, 0, 0.1)                          ]),
              make_voxel( 5, -5, 0, 1.0, [make_hit( 5, -5, 0, 0.7), make_hit( 5, -8, 0, 0.3)]),
              make_voxel( 5,  5, 0, 1.5, [make_hit( 5,  5, 0, 0.9), make_hit( 5,  8, 0, 0.6)]),
              make_voxel(10,  0, 0, 2.0, [make_hit(10,  5, 0, 1.2), make_hit(11,  5, 0, 0.8)]),
              make_voxel(15,  0, 0, 2.5, [make_hit(15,  0, 0, 1.8), make_hit(11,  0, 0, 0.7)]),
              make_voxel(20,  0, 0, 3.0, [make_hit(20,  0, 0, 1.5), make_hit(11,  0, 0, 1.5)])]

    modified_voxels, _ = drop_end_point_voxels(voxels, energy_threshold = 0.5, min_vxls = 1)
    for v in modified_voxels:
        assert np.isclose(v.E, v.hits.Ep.sum())


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
    #           x       y     z   e
    hits = [[105.0, 125.0, 77.7, 10],
            [ 95.0, 125.0, 77.7, 10],
            [ 95.0, 135.0, 77.7, 10],
            [105.0, 135.0, 77.7, 10],
            [105.0, 115.0, 77.7, 10],
            [ 95.0, 115.0, 77.7, 10],
            [ 95.0, 125.0, 79.5, 10],
            [105.0, 125.0, 79.5, 10],
            [105.0, 135.0, 79.5, 10],
            [ 95.0, 135.0, 79.5, 10],
            [ 95.0, 115.0, 79.5, 10],
            [105.0, 115.0, 79.5, 10],
            [115.0, 125.0, 79.5, 10],
            [115.0, 125.0, 85.2, 10]]
    hits = pd.DataFrame(hits, columns="X Y Z E".split())
    vox_size = np.array([15.,15.,15.],dtype=np.float16)
    voxels = voxelize_hits(hits, vox_size)
    tracks = make_track_graphs(voxels)

    assert len(tracks) == 1
    assert blob_energies(tracks[0], radius) == expected


@given(bunch_of_hits(), box_sizes, radius)
def test_blob_hits_are_inside_radius(hits, voxel_dimensions, blob_radius):
    voxels = voxelize_hits(hits, voxel_dimensions)
    tracks = make_track_graphs(voxels)
    for t in tracks:
        Ea, Eb, hits_a, hits_b   = blob_energies_and_hits(t, blob_radius)
        centre_a, centre_b       = blob_centres(t, blob_radius)

        assert all(np.linalg.norm(hits_a["X Y Z".split()] - centre_a, axis=1) < blob_radius)
        assert all(np.linalg.norm(hits_b["X Y Z".split()] - centre_b, axis=1) < blob_radius)


@given(blob_radius=radius, min_voxels=min_n_of_voxels, fraction_zero_one=fraction_zero_one)
def test_paolina_functions_with_voxels_without_associated_hits(blob_radius, min_voxels, fraction_zero_one, voxels_without_hits):
    voxels = voxels_without_hits
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

    energies = [v.E for v in voxels]
    e_thr = min(energies) + fraction_zero_one * (max(energies) - min(energies))
    mod_voxels, _ = drop_end_point_voxels(voxels, e_thr, min_voxels)

    trks = make_track_graphs(mod_voxels)
    for t in trks:
        a, b = find_extrema(t)

        assert np.allclose(blob_centre(a), a.pos)
        assert np.allclose(blob_centre(b), b.pos)


@mark.parametrize("energy_type", (HitEnergy.Ec, HitEnergy.Ep))
@given(hits                       = bunch_of_hits(),
       requested_voxel_dimensions = box_sizes,
       blob_radius                = radius,
       fraction_zero_one          = fraction_zero_one)
def test_paolina_functions_with_hit_energy_different_from_default_value(hits, requested_voxel_dimensions, blob_radius, fraction_zero_one, energy_type):
    voxels   = voxelize_hits(hits, requested_voxel_dimensions, strict_voxel_size=False)
    voxels_c = voxelize_hits(hits, requested_voxel_dimensions, strict_voxel_size=False, energy_type=energy_type)

    # The first assertion is needed for the test to keep being meaningful,
    # in case we change the default value of energy_type to energy_c.
    assert voxels[0].Etype   != voxels_c[0].Etype
    assert voxels_c[0].Etype == energy_type.value

    for voxel in voxels_c:
        assert np.isclose(voxel.E, voxel.hits[energy_type.value].sum())

    energies_c = [v.E for v in voxels_c]
    e_thr = min(energies_c) + fraction_zero_one * (max(energies_c) - min(energies_c))
    # Test that this function doesn't fail
    mod_voxels_c, _ = drop_end_point_voxels(voxels_c, e_thr, min_vxls=0)

    tot_energy     = sum(v.hits[energy_type.value].sum() for v in     voxels_c)
    tot_mod_energy = sum(v.hits[energy_type.value].sum() for v in mod_voxels_c)

    assert np.isclose(tot_energy, tot_mod_energy)

    tot_default_energy     = sum(v.hits.E.sum() for v in     voxels_c)
    tot_mod_default_energy = sum(v.hits.E.sum() for v in mod_voxels_c)

    # We don't want to modify the default energy of hits, if the voxels are made with energy_c
    if len(mod_voxels_c) < len(voxels_c):
        assert tot_default_energy > tot_mod_default_energy


def test_make_tracks_function(ICDATADIR):

    # Get some test data
    hit_file    = os.path.join(ICDATADIR, 'tracks_0000_6803_trigger2_v0.9.9_20190111_krth1600.h5')
    evt_number  = 19
    size        = 15.
    voxel_size  = np.array([size,size,size], dtype=np.float16)
    blob_radius = 21*units.mm

    # Read the hits and voxelize
    all_hits = pd.read_hdf(hit_file, "/RECO/Events")

    for evt_number, evt_hits in all_hits.groupby("event"):
        evt_time = evt_hits.time.iloc[0]
        voxels   = voxelize_hits(evt_hits, voxel_size, strict_voxel_size=False, energy_type=HitEnergy.E)

        tracks   = list(make_track_graphs(voxels))

        track_coll = make_tracks(evt_number, evt_time, voxels, voxel_size,
                                 contiguity=Contiguity.CORNER,
                                 blob_radius=blob_radius,
                                 energy_type=HitEnergy.E)
        tracks_from_coll = track_coll.tracks

        tracks.sort          (key=lambda x : len(x.nodes()))
        tracks_from_coll.sort(key=lambda x : x.number_of_voxels)

        # Compare the two sets of tracks
        assert len(tracks) == len(tracks_from_coll)
        for i in range(len(tracks)):
            t  = tracks[i]
            tc = tracks_from_coll[i]

            assert len(t.nodes())              == tc.number_of_voxels
            assert sum(v.E for v in t.nodes()) == tc.E

            tc_blobs = list(tc.blobs)
            tc_blobs.sort(key=lambda x : x.E)
            tc_blob_energies = (tc.blobs[0].E, tc.blobs[1].E)

            assert np.allclose(blob_energies(t, blob_radius), tc_blob_energies)


@given(bunch_of_hits(), box_sizes)
def test_make_voxel_graph_keeps_energy_consistence(hits, voxel_dimensions):
    voxels = voxelize_hits    (hits  , voxel_dimensions)
    tracks = make_track_graphs(voxels)
    # assert sum of track energy equal to sum of hits energies
    assert_almost_equal(sum(get_track_energy(track) for track in tracks), hits.E.sum())
