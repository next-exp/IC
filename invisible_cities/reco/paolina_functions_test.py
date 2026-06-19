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
from hypothesis            import HealthCheck
from hypothesis.strategies import composite
from hypothesis.strategies import lists
from hypothesis.strategies import floats
from hypothesis.strategies import integers
from hypothesis.strategies import builds

from . paolina_functions import blob_energies_hits_and_centres, round_hits_positions_in_place
from . paolina_functions import voxelize_hits
from . paolina_functions import neighbours
from . paolina_functions import find_extrema_and_length
from . paolina_functions import hits_ave_pos
from . paolina_functions import shortest_paths
from . paolina_functions import assign_blobs_inplace
from . paolina_functions import pop_voxel_inplace
from . paolina_functions import make_track_graphs
from . paolina_functions import drop_voxels
from . paolina_functions import make_tracks

from .. core                import system_of_units as units
from .. core.core_functions import in_range
from .. core.exceptions     import NoHits
from .. core.exceptions     import NoVoxels
from .. core.testing_utils  import assert_dataframes_close
from .. core.testing_utils  import an_instance_of

from .. io.mcinfo_io import load_mchits_df

from .. types.symbols  import Contiguity
from .. types.symbols  import HitEnergy


def spread_enough(hits):
    xyz  = list("XYZ")
    low  = hits[xyz].min()
    high = hits[xyz].max()
    return np.all(high-low > 0.1)


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
    assume(spread_enough(hits))
    return hits


@composite
def single_voxels(draw, vox_size=np.array([15.55, 15.55, 4.0])):
    x = draw(integers(-15, 15))
    y = draw(integers(-15, 15))
    z = draw(integers(  0, 30))

    return pd.Series(dict( x = x * vox_size[0]
                         , y = y * vox_size[1]
                         , z = z * vox_size[2]
                         , e = draw(floats(1.0, 234.0))
                         ), name=hash((x,y,z)))


@composite
def bunch_of_voxels(draw):
    voxel_size = np.array([draw(floats(1,10)) for _ in range(3)])
    a_voxel    = single_voxels(voxel_size).map(lambda s: s.to_frame().T)
    strat      = lists(a_voxel, min_size=1, max_size=30)

    voxels = draw(strat)
    voxels = pd.concat(voxels)
    assume(voxels.index.nunique() == len(voxels))
    return voxels, voxel_size


voxel_sizes = builds( np.array
                    , lists(floats(min_value=1, max_value=5),
                            min_size = 3,
                            max_size = 3))
radii             =   floats(min_value=1, max_value=100)
min_n_of_voxels   = integers(min_value=3, max_value= 10)
fraction_zero_one =   floats(min_value=0, max_value=  1)


@composite
def hits_and_voxels(draw):
    hits       = draw(bunch_of_hits())
    voxel_size = draw(voxel_sizes)
    voxels     = voxelize_hits(hits, voxel_size)
    return hits, voxels, voxel_size


@given(bunch_of_hits())
def test_round_hits_positions_in_place(hits):
    """
    Override xyz such that all values fall below the rounding decimal place. We
    also multiply some values by -1 to include negative numbers. The maximum
    absolute value xyz can have is 100. After multiplying by 1e-7, the maximum
    absolute value is 1e-5. After rounding, the only possible values are 0, 1e-5
    or -1e-5.
    """
    xyz=list("XYZ")
    hits.loc[:, xyz] = hits[xyz].values * 0.999e-7 * [-1, 1, -1]

    round_hits_positions_in_place(hits, 5)

    assert np.all(np.in1d(hits[xyz], [0, 1e-5, -1e-5]))


@an_instance_of(single_hits())
def test_round_hits_positions_in_place_empty_input(hits):
    """
    It simply should not crash.
    """
    hits = hits.iloc[:0]
    round_hits_positions_in_place(hits, 5)
    assert len(hits) == 0


@an_instance_of(single_hits())
def test_round_hits_positions_in_place_non_finite_values(hit):
    """
    Override xyz with np.nan and np.inf, ensure values are not changed.
    """
    hit.loc[:, "X"] =  np.nan
    hit.loc[:, "Y"] =  np.inf
    hit.loc[:, "Z"] = -np.inf


    round_hits_positions_in_place(hit, 5)
    unchanged = np.isclose( hit["X Y Z".split()].values[0]
                          , np.array([np.nan, np.inf, -np.inf])
                          , equal_nan=True)
    assert np.all(unchanged)


@an_instance_of(single_hits(), voxel_sizes)
def test_voxelize_hits_raises_if_no_hits(hits, voxel_size):
    empty = hits.iloc[:0]
    with raises(NoHits):
        voxelize_hits(empty, voxel_size)


@given(bunch_of_hits(), voxel_sizes)
def test_voxelize_hits_does_not_lose_energy(hits, voxel_size):
    hits, voxels  = voxelize_hits(hits, voxel_size, energy_type=HitEnergy.E)
    total_voxel_e = voxels.e.sum()
    total_hits_e  = hits.E.sum()

    assert np.isclose(total_voxel_e, total_hits_e)


@parametrize("hit_energy", HitEnergy)
@given(hits=bunch_of_hits(), voxel_size=voxel_sizes)
def test_voxelize_hits_voxel_energy_equals_hits_energy(hits, voxel_size, hit_energy):
    hits, voxels  = voxelize_hits(hits, voxel_size, energy_type=hit_energy)
    for vid, hs in hits.groupby("voxel_id"):
        hits_e = hs[hit_energy.value].sum()
        assert np.isclose(hits_e, voxels.loc[vid, "e"])


@given(bunch_of_hits(), voxel_sizes)
def test_voxelize_hits_within_hits_limits(hits, voxel_size):
    hits, voxels = voxelize_hits(hits, voxel_size, HitEnergy.E)

    xyz = list("xyz")
    XYZ = list("XYZ")
    assert np.all(hits[XYZ].min().values <= voxels[xyz].min().values + voxel_size/2)
    assert np.all(hits[XYZ].max().values >= voxels[xyz].max().values - voxel_size/2)


@given(bunch_of_hits(), voxel_sizes)
def test_voxelize_hits_respects_voxel_size(hits, voxel_size):
    _, voxels = voxelize_hits(hits, voxel_size, HitEnergy.E)

    xyz = list("xyz")
    for xyz1, xyz2 in combinations(voxels[xyz].values, 2):
        distance_between_voxels = xyz2 - xyz1
        off_by = distance_between_voxels % voxel_size
        assert (np.isclose(off_by, 0         ) |
                np.isclose(off_by, voxel_size) ).all()


def test_voxelize_hits_border_are_assigned_to_correct_voxel():
    hits   = pd.DataFrame(dict( X = [ 5, 15, 15, 15, 25]
                              , Y = [15,  5, 15, 25, 15]
                              , Z = [ 5,  5,  5, 15, 15]
                              , E = 1 # dummy
                              ), dtype=float)

    vox_size = np.ones(3) * 10.0
    hits, voxels = voxelize_hits(hits, vox_size, HitEnergy.E)

    voxel_indices = ( (0,1,0), (1,0,0), (1,1,0), (1,2,1), (2,1,1) )
    assert len(voxels) == 5
    assert hits.voxel_id.to_list() == list(map(hash, voxel_indices))


@an_instance_of(single_hits())
def test_voxelize_single_hit(hit):
    vox_size = np.ones(3) * 10.
    _, voxels = voxelize_hits(hit, vox_size, HitEnergy.E)
    assert len(voxels) == 1


def test_neighbours_corner():
    v1 = pd.Series(dict(x=0, y=0, z=0), dtype=float)
    v2 = pd.Series(dict(x=1, y=0, z=0), dtype=float)
    v3 = pd.Series(dict(x=1, y=1, z=0), dtype=float)
    v4 = pd.Series(dict(x=1, y=1, z=1), dtype=float)

    voxel_size = np.ones(3, dtype=float)
    assert     neighbours(v1, v2, voxel_size, Contiguity.FACE)
    assert not neighbours(v1, v3, voxel_size, Contiguity.FACE)
    assert not neighbours(v1, v4, voxel_size, Contiguity.FACE)

    assert     neighbours(v1, v2, voxel_size, Contiguity.EDGE)
    assert     neighbours(v1, v3, voxel_size, Contiguity.EDGE)
    assert not neighbours(v1, v4, voxel_size, Contiguity.EDGE)

    assert     neighbours(v1, v2, voxel_size, Contiguity.CORNER)
    assert     neighbours(v1, v3, voxel_size, Contiguity.CORNER)
    assert     neighbours(v1, v4, voxel_size, Contiguity.CORNER)


@given(bunch_of_voxels())
def test_make_track_graphs_keeps_all_voxels(voxels_and_size):
    voxels, voxel_size = voxels_and_size
    tracks = make_track_graphs(voxels, voxel_size, Contiguity.CORNER)
    n_voxels_in_tracks = sum(len(t.nodes()) for t in tracks)
    assert n_voxels_in_tracks == len(voxels)


@given(bunch_of_voxels())
def test_make_track_graphs_voxels_are_exclusive(voxels_and_size):
    voxels, voxel_size = voxels_and_size
    tracks = make_track_graphs(voxels, voxel_size, Contiguity.CORNER)
    for t1, t2 in combinations(tracks, 2):
        assert set(t1.nodes).intersection(set(t2.nodes)) == set()


FACE, EDGE, CORNER = Contiguity
@parametrize('contiguity,  are_neighbours,        voxels',
             ((FACE,            True     , [[0,0,0], [0,0,1]]), # share face
              (FACE,            False    , [[0,0,0], [0,1,1]]), # share edge
              (FACE,            False    , [[0,0,0], [1,1,1]]), # share corner
              (FACE,            False    , [[0,0,0], [2,2,2]]), # share nothing
              (FACE,            False    , [[0,0,0], [2,0,0]]), # share nothing, but aligned

              (EDGE,            True     , [[0,0,0], [0,0,1]]), # share face
              (EDGE,            True     , [[0,0,0], [0,1,1]]), # share edge
              (EDGE,            False    , [[0,0,0], [1,1,1]]), # share corner
              (EDGE,            False    , [[0,0,0], [2,2,2]]), # share nothing
              (EDGE,            False    , [[0,0,0], [2,0,0]]), # share nothing, but aligned

              (CORNER,          True     , [[0,0,0], [0,0,1]]), # share face
              (CORNER,          True     , [[0,0,0], [0,1,1]]), # share edge
              (CORNER,          True     , [[0,0,0], [1,1,1]]), # share corner
              (CORNER,          False    , [[0,0,0], [2,2,2]]), # share nothing
              (CORNER,          False    , [[0,0,0], [2,0,0]]), # share nothing, but aligned
             ))
def test_make_track_graphs_contiguity(contiguity, are_neighbours, voxels):
    voxels = pd.DataFrame(voxels, columns=list("xyz")).assign(e=np.arange(len(voxels)))
    voxel_size = np.ones(3, dtype=float)

    expected_number_of_tracks = 1 if are_neighbours else 2
    tracks = make_track_graphs(voxels, voxel_size, contiguity=contiguity)

    assert len(tracks) == expected_number_of_tracks


def test_shortest_paths_all_connected():
    nodes = list(range(10))
    g = nx.Graph()
    g.add_nodes_from(nodes)
    for i, ni in enumerate(nodes[:-1]):
        for nj in nodes[i+1:]:
            g.add_edge(ni, nj, distance=7)
    distances = shortest_paths(g)
    assert np.all(distances.distance.isin([0,7]))


def test_shortest_paths_all_linear():
    nodes = list(range(10))
    g = nx.Graph()
    g.add_nodes_from(nodes)
    for i, ni in enumerate(nodes[:-1]):
        g.add_edge(ni, nodes[i+1], distance=7)
    distances = shortest_paths(g)
    assert np.all(distances.distance % 7 == 0)
    assert np.all(distances.distance == np.abs(distances.final - distances.initial) * 7)


@an_instance_of(bunch_of_voxels())
def test_find_extrema_and_length_single_voxel(voxels_and_size):
    voxels, _ = voxels_and_size
    voxels = voxels.iloc[:1]
    index  = voxels.index[0]
    g = nx.Graph()
    g.add_node(index)
    assert find_extrema_and_length(g, voxels) == (index, index, 0.)


def test_find_extrema_and_length_no_voxels():
    dummy = pd.DataFrame()
    with raises(NoVoxels):
        find_extrema_and_length({}, dummy)


@fixture(scope='module')
def voxels_without_hits():
    voxels = pd.DataFrame([[10,10,10, 1],
                           [10,10,11, 2],
                           [10,10,12, 3],
                           [10,10,13, 4],
                           [10,10,14, 5],
                           [10,10,15, 6],
                           [10,11,15, 7],
                           [10,12,15, 8],
                           [10,13,15, 9],
                           [10,14,15,10],
                           [10,15,15,11]], columns=list("xyze"))
    return voxels


@fixture
def linear_track_face():
    t          = np.arange(10)
    hits       = pd.DataFrame(dict(X=t, Y=0, Z=0, E=t+1, voxel_id=t))
    voxel_size = np.ones(3, dtype=float)
    distance   = 1
    voxels     = hits.groupby("voxel_id").first().rename(columns={a: a.lower() for a in "XYZE"})
    graph      = nx.Graph()
    graph.add_nodes_from(t)
    for i in range(len(t)-1):
        graph.add_edge(i, i+1, distance=distance)
    return hits, voxels, voxel_size, distance, graph


@fixture
def linear_track_edge():
    t          = np.arange(10)
    hits       = pd.DataFrame(dict(X=t, Y=t, Z=0, E=t+1, voxel_id=t))
    voxel_size = np.ones(3, dtype=float)
    distance   = 2**0.5
    voxels     = hits.groupby("voxel_id").first().rename(columns={a: a.lower() for a in "XYZE"})
    graph      = nx.Graph()
    graph.add_nodes_from(t)
    for i in range(len(t)-1):
        graph.add_edge(i, i+1, distance=distance)
    return hits, voxels, voxel_size, distance, graph


@fixture
def linear_track_corner():
    t          = np.arange(10)
    hits       = pd.DataFrame(dict(X=t, Y=t, Z=t, E=t+1, voxel_id=t))
    voxel_size = np.ones(3, dtype=float)
    distance   = 3**0.5
    voxels     = hits.groupby("voxel_id").first().rename(columns={a: a.lower() for a in "XYZE"})
    graph      = nx.Graph()
    graph.add_nodes_from(t)
    for i in range(len(t)-1):
        graph.add_edge(i, i+1, distance=distance)
    return hits, voxels, voxel_size, distance, graph


@fixture
def pseudo_nonlinear_track():
    # a track that connects all pairs of voxels, but the distance between them
    # is proportional to their position in the sequence
    t          = np.arange(10)
    hits       = pd.DataFrame(dict(X=t, Y=t, Z=t, E=t+1, voxel_id=t))
    voxel_size = np.ones(3, dtype=float)
    voxels     = hits.groupby("voxel_id").first().rename(columns={a: a.lower() for a in "XYZE"})
    graph      = nx.Graph()
    graph.add_nodes_from(t)
    for i in range(len(t)-1):
        for j in range(i+1, len(t)):
            graph.add_edge(i, j, distance = j-i)
    return hits, voxels, voxel_size, graph


@fixture(params=[0,1,2])
def linear_tracks(request, linear_track_face, linear_track_edge, linear_track_corner):
    tracks = [linear_track_face, linear_track_edge, linear_track_corner]
    return tracks[request.param]


@fixture
def ushaped_track():
    t          = np.arange(10)
    hits       = pd.concat( [ pd.DataFrame(dict(X=0, Y=0, Z=t      , E=t+ 1, voxel_id=t    ))
                            , pd.DataFrame(dict(X=1, Y=1, Z=[10]   , E=11  , voxel_id=10   ))
                            , pd.DataFrame(dict(X=2, Y=2, Z=t[::-1], E=t+12, voxel_id=t+11 ))
                            ])
    voxel_size = np.ones(3, dtype=float)
    voxels     = hits.groupby("voxel_id").first().rename(columns={a: a.lower() for a in "XYZE"})
    graph      = nx.Graph()
    graph.add_nodes_from(t)
    for i in range(len(hits)-1):
        graph.add_edge(i, i+1, distance = 3**0.5 if i in [9, 10] else 1)
    return hits, voxels, voxel_size, graph


def test_find_extrema_and_length_linear(linear_tracks):
    _, voxels, _, distance, graph = linear_tracks

    v1, v2, length = find_extrema_and_length(graph, voxels)

    assert v1 == 0
    assert v2 == len(voxels) - 1
    assert np.isclose(length, (len(voxels) - 1) * distance)


def test_find_extrema_and_length_pseudo_nonlinear(pseudo_nonlinear_track):
    _, voxels, _, graph = pseudo_nonlinear_track

    v1, v2, length = find_extrema_and_length(graph, voxels)

    assert v1 == 0
    assert v2 == len(voxels) - 1
    assert length == (len(voxels) - 1)


def test_find_extrema_and_length_ushaped(ushaped_track):
    _, voxels, _, graph = ushaped_track

    v1, v2, length = find_extrema_and_length(graph, voxels)

    assert v1 == 0
    assert v2 == len(voxels) - 1
    assert length == (len(voxels) - 3) + 2*3**0.5


@parametrize('contiguity, expected_length',
             ((Contiguity.FACE  , 4),
              (Contiguity.CORNER, 2 * sqrt(2))
             ))
def test_find_extrema_and_length_around_bend(contiguity, expected_length):
    # Make sure that we calculate the length along the track rather
    # that the shortcut
    voxels = pd.DataFrame([[0,0,0,0],
                           [1,0,0,1],
                           [1,1,0,2],
                           [1,2,0,3],
                           [0,2,0,4]], columns=list("xyze"), dtype=float)
    voxel_size = np.ones(3, dtype=float)

    tracks = make_track_graphs(voxels, voxel_size, contiguity=contiguity)
    assert len(tracks) == 1

    graph = tracks[0]
    v1, v2, length = find_extrema_and_length(graph, voxels)

    assert v1 == 0
    assert v2 == 4
    assert np.isclose(length, expected_length)


@parametrize('contiguity expected_length'.split(),
             (
              (Contiguity.FACE  , 1 + 1 + 1  ), # Face contiguity requires 3 steps, each parallel to an axis
              (Contiguity.EDGE  , 1 + sqrt(2)), # Edge continuity allows to cut one corner
              (Contiguity.CORNER,     sqrt(3)), # Corner contiguity makes it possible to do in a single step
             ))
def test_find_extrema_and_length_cuts_corners(contiguity, expected_length):
    "Make sure that we cut corners, if the contiguity allows"
    voxels = pd.DataFrame([[0,0,0,1], # Extremum 1
                           [1,0,0,2],
                           [1,1,0,3],
                           [1,1,1,4]] # Extremum 2
                         , columns=list("xyze"), dtype=float)
    voxel_size = np.ones(3, dtype=float)

    tracks = make_track_graphs(voxels, voxel_size, contiguity=contiguity)
    assert len(tracks) == 1

    graph = tracks[0]
    v1, v2, length = find_extrema_and_length(graph, voxels)

    assert v1 == 0
    assert v2 == 3
    assert np.isclose(length, expected_length)


def test_hits_ave_pos():
    hits = pd.DataFrame([[0,0,1,1],
                         [1,1,0,2],
                         [0,1,1,3]]
                        , columns="X Y Z E".split())
    ave_pos = hits_ave_pos(hits)
    expected = np.array([2/6, 5/6, 4/6])
    assert np.allclose(ave_pos, expected)


def test_assign_blobs_inplace_linear(linear_tracks):
    hits, voxels, voxel_size, _, graph = linear_tracks
    n       = len(voxels) # == len(hits)
    hits    = hits  .assign(blob="none")
    voxels  = voxels.assign(blob="none")

    # very small, blob radius, so only one voxel is in the blob
    assign_blobs_inplace(graph, hits, voxels, 0.1, 0, n-1, voxel_size)

    assert len(hits  ) == n
    assert len(voxels) == n
    assert    hits.blob.iloc[   0]        == "low"
    assert    hits.blob.iloc[  -1]        == "high"
    assert (  hits.blob.iloc[1:-1].values == "none").all()
    assert  voxels.blob.iloc[   0]        == "low"
    assert  voxels.blob.iloc[  -1]        == "high"
    assert (voxels.blob.iloc[1:-1].values == "none").all()


def test_assign_blobs_inplace_multivoxel(linear_track_face):
    hits, voxels, voxel_size, _, graph = linear_track_face
    n       = len(voxels) # == len(hits)
    hits    = hits  .assign(blob="none")
    voxels  = voxels.assign(blob="none")

    # a larger blob radius, so more than one voxel is in the blob
    assign_blobs_inplace(graph, hits, voxels, 1.1, 0, n-1, voxel_size)

    assert len(hits  ) == n
    assert len(voxels) == n
    assert (  hits.blob.iloc[  :2]        == "low" ).all()
    assert (  hits.blob.iloc[-2: ]        == "high").all()
    assert (  hits.blob.iloc[2:-2].values == "none").all()
    assert (voxels.blob.iloc[  :2]        == "low" ).all()
    assert (voxels.blob.iloc[-2: ]        == "high").all()
    assert (voxels.blob.iloc[2:-2].values == "none").all()


def test_assign_blobs_inplace_ushaped(ushaped_track):
    hits, voxels, voxel_size, graph = ushaped_track
    n       = len(voxels) # == len(hits)
    hits    = hits  .assign(blob="none")
    voxels  = voxels.assign(blob="none")

    # a blob radius slightly larger than the distance between the two extremes
    # so that we ensure it doesn't incorrectly pick up the other extreme despite
    # being close. Extremes are at a distance of sqrt(8) = 2.83, so we choose
    # 2.9<3 which should include exactly three voxels from each end
    assign_blobs_inplace(graph, hits, voxels, 2.9, 0, n-1, voxel_size)

    assert len(hits  ) == n
    assert len(voxels) == n
    assert (  hits.blob.iloc[  :3]        == "low" ).all()
    assert (  hits.blob.iloc[-3: ]        == "high").all()
    assert (  hits.blob.iloc[3:-3].values == "none").all()
    assert (voxels.blob.iloc[  :3]        == "low" ).all()
    assert (voxels.blob.iloc[-3: ]        == "high").all()
    assert (voxels.blob.iloc[3:-3].values == "none").all()


@parametrize('radius low_e high_e'.split(),
             ((10.,  20,  40),
              (12.,  40,  80),
              (14.,  40,  80),
              (16.,  80,  80),
              (18.,  80,  80),
              (20.,  80, 100),
              (22., 100, 140)
 ))

def test_blobs(radius, low_e, high_e):
    #           x       y     z   e  event
    hits = [[105.0, 125.0, 77.7, 10, 0],
            [ 95.0, 125.0, 77.7, 10, 0],
            [ 95.0, 135.0, 77.7, 10, 0],
            [105.0, 135.0, 77.7, 10, 0],
            [105.0, 115.0, 77.7, 10, 0],
            [ 95.0, 115.0, 77.7, 10, 0],
            [ 95.0, 125.0, 79.5, 10, 0],
            [105.0, 125.0, 79.5, 10, 0],
            [105.0, 135.0, 79.5, 10, 0],
            [ 95.0, 135.0, 79.5, 10, 0],
            [ 95.0, 115.0, 79.5, 10, 0],
            [105.0, 115.0, 79.5, 10, 0],
            [115.0, 125.0, 79.5, 10, 0],
            [115.0, 125.0, 85.2, 10, 0]]
    hits = pd.DataFrame(hits, columns="X Y Z E event".split())
    hits['Ep'] = hits['E']

    voxel_size           = np.array([15.,15.,15.],dtype=float)
    hits, voxels         = voxelize_hits(hits, voxel_size, HitEnergy.E)
    hits, voxels, tracks = make_tracks(hits, voxels, voxel_size, radius, Contiguity.CORNER, HitEnergy.E)

    assert   hits.track.nunique() == 1
    assert voxels.track.nunique() == 1

    assert np.isclose(hits.loc[hits.blob.str.contains("low" )].E.sum(),  low_e)
    assert np.isclose(hits.loc[hits.blob.str.contains("high")].E.sum(), high_e)


@settings(deadline=None)
@given(bunch_of_hits(), voxel_sizes, radii)
def test_make_tracks_blob_hits_are_inside_radius(hits, voxel_size, blob_radius):
    hits, voxels         = voxelize_hits(hits, voxel_size, HitEnergy.E)
    hits, voxels, tracks = make_tracks(hits, voxels, voxel_size, blob_radius, Contiguity.CORNER, HitEnergy.E)

    diag = np.linalg.norm(voxel_size)
    for t in voxels.track.unique():
        hits_track   =   hits.loc[  hits.track == t]
        voxels_track = voxels.loc[voxels.track == t]
        hits_low  = hits_track.loc[hits_track.blob.str.contains("low" )]
        hits_high = hits_track.loc[hits_track.blob.str.contains("high")]

        centre_low  = hits_ave_pos(hits_low )
        centre_high = hits_ave_pos(hits_high)

        xyz = list("XYZ")
        assert all(np.linalg.norm(hits_low [xyz] - centre_low , axis=1) < blob_radius + diag)
        assert all(np.linalg.norm(hits_high[xyz] - centre_high, axis=1) < blob_radius + diag)


@fixture
def dummy_voxels():
    n = 10
    t = np.arange(n)
    voxels = pd.DataFrame(dict(x=t+1, y=t+2, z=t+3, e=t+4), index=t)
    return voxels, n


def test_pop_voxel_inplace(dummy_voxels):
    voxels, n = dummy_voxels
    k =  n//2

    popped = pop_voxel_inplace(voxels, k)
    assert len(voxels) == n-1
    assert k not in voxels.index
    assert popped.name == k


def test_pop_voxel_inplace_raises_if_missing(dummy_voxels):
    voxels, n = dummy_voxels
    k = n + 1

    with raises(KeyError):
        pop_voxel_inplace(voxels, k)


@settings(deadline=None)
@given(bunch_of_hits(), voxel_sizes, min_n_of_voxels, fraction_zero_one)
def test_energy_is_conserved_with_dropped_voxels(hits, requested_voxel_size, min_voxels, fraction_zero_one):
    tot_initial_energy = hits.E.sum()
    hits, voxels               = voxelize_hits(hits, requested_voxel_size)

    # tracks before dropping (we don't care about blob radius)
    i_hits, i_voxels, i_tracks = make_tracks(hits, voxels, requested_voxel_size, 1)
    i_Es                       = np.sort(i_tracks.energy.values)

    # set a threshold
    energies = voxels.e.values
    e_thr = min(energies) + fraction_zero_one * (max(energies) - min(energies))

    # drop voxels and reconstruct tracks
    d_hits, d_voxels, d_dropped = drop_voxels(hits, voxels, e_thr, requested_voxel_size, HitEnergy.E)
    f_hits, f_voxels, f_tracks  = make_tracks(hits, voxels, requested_voxel_size, 1)
    f_Es                        = np.sort(f_tracks.energy.values)
    tot_final_energy            = f_hits.E.sum()

    assert tot_initial_energy == approx(tot_final_energy)
    assert np.allclose(i_Es, f_Es)


@settings(deadline=None)
@mark.parametrize("energy_type", HitEnergy)
@given(hits                       = bunch_of_hits(),
       requested_voxel_size = voxel_sizes,
       min_voxels                 = min_n_of_voxels,
       fraction_zero_one          = fraction_zero_one)
def test_dropped_voxels_have_nan_energy(hits, requested_voxel_size, min_voxels, fraction_zero_one, energy_type):

    hits, voxels = voxelize_hits(hits, requested_voxel_size)
    # if the tracks are too short, there are no dropped voxels
    assume(len(voxels) >= min_voxels)
    energies     = voxels.e.values
    e_thr        = min(energies) + fraction_zero_one * (max(energies) - min(energies))

    d_hits, d_voxels, d_dropped = drop_voxels(hits, voxels, e_thr, requested_voxel_size, energy_type, min_voxels)

    if not d_dropped.empty:
        assert np.all(np.isnan(d_dropped.e))
        assert np.all(np.isnan(d_hits[energy_type.value]))


@settings(deadline=None)
@mark.parametrize("energy_type", HitEnergy)
@given(hits                       = bunch_of_hits(),
       requested_voxel_size = voxel_sizes,
       min_voxels                 = min_n_of_voxels,
       fraction_zero_one          = fraction_zero_one)
def test_drop_end_point_voxels_doesnt_modify_other_energy_types(hits, requested_voxel_size, min_voxels, fraction_zero_one, energy_type):
    hits, voxels                  = voxelize_hits(hits, requested_voxel_size)
    energies                      = voxels.e.values
    e_thr                         = min(energies) + fraction_zero_one * (max(energies) - min(energies))
    d_hits, d_voxels, d_dropped   = drop_voxels(hits, voxels, e_thr, requested_voxel_size, energy_type, min_voxels)
    # collect all hits from dropped voxels that you want to check
    i_hits = hits.loc[hits.index.isin(d_hits.index)]

    for e_type in HitEnergy:
        if e_type is energy_type: continue
        if d_hits.empty:
            return

        e_before = i_hits[e_type.value]
        e_after  = d_hits[e_type.value]
        assert np.allclose(e_before.values, e_after.values)


@mark.parametrize("energy_type", HitEnergy)
@given(hits                       = bunch_of_hits(),
       requested_voxel_size = voxel_sizes,
       min_voxels                 = min_n_of_voxels,
       fraction_zero_one          = fraction_zero_one)
def test_drop_voxels_voxel_energy_is_sum_of_hits_general(hits, requested_voxel_size, min_voxels, fraction_zero_one, energy_type):
    hits, voxels                  = voxelize_hits(hits, requested_voxel_size, energy_type)
    energies                      = voxels.e.values
    e_thr                         = min(energies) + fraction_zero_one * (max(energies) - min(energies))
    d_hits, d_voxels, d_dropped   = drop_voxels(hits, voxels, e_thr, requested_voxel_size, energy_type, min_voxels)

    for idx, row in voxels.iterrows():
        assert row.e == (hits[hits.voxel_id == idx])[energy_type.value].sum()


@settings(suppress_health_check=[HealthCheck.filter_too_much, HealthCheck.too_slow], deadline=None)
@mark.parametrize("energy_type", HitEnergy)
@given(hits                       = bunch_of_hits(),
       requested_voxel_size = voxel_sizes,
       min_voxels                 = min_n_of_voxels,
       fraction_zero_one          = fraction_zero_one)
def test_drop_end_point_voxels_constant_number_of_voxels_and_hits(hits, requested_voxel_size, min_voxels, fraction_zero_one, energy_type):
    hits, voxels                  = voxelize_hits(hits, requested_voxel_size, energy_type)
    assume(len(voxels) >= min_voxels)
    i_hit_len = len(hits)
    i_vox_len = len(voxels)
    energies                      = voxels.e.values
    e_thr                         = min(energies) + fraction_zero_one * (max(energies) - min(energies))
    d_hits, d_voxels, d_dropped   = drop_voxels(hits, voxels, e_thr, requested_voxel_size, energy_type, min_voxels)

    assume(not d_dropped.empty)
    assert (len(d_voxels) + len(d_dropped)) == i_vox_len
    # hits aren't dropped
    assert len(hits)   == i_hit_len


def test_initial_voxels_are_modified_inplace_after_dropping_voxels(ICDATADIR):
    # prior test ensured voxels are the same (not edited in-place)
    # we're changing that, so this test has been reworked

    # Get some test data: nothing interesting to see here
    hit_file = os.path.join(ICDATADIR, 'tracks_0000_6803_trigger2_v0.9.9_20190111_krth1600.h5')
    evt_number = 19
    e_thr = 5867.92
    min_voxels = 3
    size = 15.
    energy_type = HitEnergy.E
    vox_size = np.array([size,size,size], dtype=np.float16)
    hits = pd.read_hdf(hit_file, "/RECO/Events")
    hits = hits[hits.event == evt_number]

    hits, voxels                  = voxelize_hits(hits, vox_size, energy_type)

    i_energies  = np.sort(voxels.e.values)
    i_positions = np.sort([voxels.x, voxels.y, voxels.z])

    d_hits, d_voxels, d_dropped   = drop_voxels(hits, voxels, e_thr, vox_size, energy_type, min_voxels)

    # energies will be retained (in principle)
    f_energies  = np.sort(voxels.e.values)
    f_positions = np.sort(np.concatenate([[voxels.x, voxels.y, voxels.z], [d_dropped.x, d_dropped.y, d_dropped.z]], axis = 1))

    # energies match
    assert sum(f_energies) == sum(i_energies)
    # but lengths do not
    assert len(f_energies) != len(i_energies)
    # voxels haven't disappeared, just reindexed
    assert len(f_positions) == len(i_positions)
    assert np.allclose(f_positions, i_positions)


def test_tracks_with_dropped_voxels(ICDATADIR):
    # Get some test data: nothing interesting to see here
    hit_file = os.path.join(ICDATADIR, 'tracks_0000_6803_trigger2_v0.9.9_20190111_krth1600.h5')
    evt_number = 19
    e_thr = 5867.92
    min_voxels = 3
    size = 15.
    energy_type = HitEnergy.E
    vox_size = np.array([size,size,size], dtype=np.float16)
    hits = pd.read_hdf(hit_file, "/RECO/Events")
    hits = hits[hits.event == evt_number]
    hits, voxels                  = voxelize_hits(hits, vox_size, energy_type)

    _, _, i_tracks    = make_tracks(hits, voxels, vox_size, 1)
    i_energies        = np.sort(i_tracks.energy.values)
    i_ntracks         = len(i_tracks)
    i_nvoxels         = i_tracks.numb_of_voxels.values

    d_hits, d_voxels, d_dropped = drop_voxels(hits, voxels, e_thr, vox_size, energy_type, min_voxels)

    _, _, f_tracks    = make_tracks(hits, voxels, vox_size, 1)
    f_energies        = np.sort(f_tracks.energy.values)
    f_ntracks         = len(f_tracks)
    f_nvoxels         = f_tracks.numb_of_voxels.values


    expected_diff_nvoxels = np.array([3, 0, 0])


    assert i_ntracks == f_ntracks
    assert np.allclose(i_energies, f_energies)
    assert np.all(i_nvoxels - f_nvoxels == expected_diff_nvoxels)


def test_drop_voxels_deterministic(ICDATADIR):
    hit_file   = os.path.join(ICDATADIR, 'tracks_0000_6803_trigger2_v0.9.9_20190111_krth1600.h5')
    evt_number = 19
    e_thr      = 5867.92
    min_voxels = 3
    vox_size   = [15.] * 3
    energy_type = HitEnergy.E

    hits = pd.read_hdf(hit_file, "/RECO/Events")
    hits = hits[hits.event == evt_number]

    hits, voxels                  = voxelize_hits(hits, vox_size, energy_type)
    # to avoid in place issues
    c_hits   = hits.copy(deep=True)
    c_voxels = voxels.copy(deep=True)
    d_hits, d_voxels, d_dropped = drop_voxels(c_hits, c_voxels, e_thr, vox_size, energy_type, min_voxels)
    # shuffle hits and voxels
    hits   = hits.sample(frac=1).reset_index(drop=True)
    voxels = voxels.sample(frac=1)
    r_hits, r_voxels, r_dropped = drop_voxels(hits, voxels, e_thr, vox_size, energy_type, min_voxels)

    assert np.allclose(np.sort(r_voxels.e.values), np.sort(voxels.e.values))


def test_voxel_drop_in_short_tracks():
    hits         = pd.DataFrame(dict(X=[10, 26], Y=[10,10], Z=[10,10], E=[1,1]))
    energy_type  = HitEnergy.E
    vox_size     = [15.] * 3
    min_voxels = 0

    hits, voxels = voxelize_hits(hits, vox_size, energy_type)
    hits = hits.copy()
    e_thr        = sum(voxels.e) + 1.

    d_hits, d_voxels, d_dropped = drop_voxels(hits, voxels, e_thr, vox_size, energy_type, min_voxels)

    assert len(voxels) >= 1


def test_drop_voxels_voxel_energy_is_sum_of_hits():

    x = [0,     5,   5,  10,  15,  20,   5,   5,  11,  11,  11]
    y = [0,    -5,   5,   5,   0,   0,  -8,   8,   5,   0,   0]
    z = [0,     0,   0,   0,   0,   0,   0,   0,   0,   0,   0]
    e = [0.1, 0.7, 0.9, 1.2, 1.8, 1.5, 0.3, 0.6, 0.8, 0.7, 1.5]
    hits = pd.DataFrame(dict(X=x, Y=y, Z=z, E=e))

    vox_size = [5.]*3
    hits, voxels = voxelize_hits(hits, vox_size, HitEnergy.E)

    assert np.isclose(hits.groupby('voxel_id').E.sum(), voxels.e).all()


@settings(deadline=None)
@mark.parametrize("energy_type", (HitEnergy.Ec, HitEnergy.Ep))
@given(hits                       = bunch_of_hits(),
       requested_voxel_size = voxel_sizes,
       blob_radius                = radii,
       fraction_zero_one          = fraction_zero_one)
def test_paolina_functions_with_hit_energy_different_from_default_value(hits, requested_voxel_size, blob_radius, fraction_zero_one, energy_type):

    hits_c     = hits.copy(deep = True)
    min_voxels = 0

    hits, voxels     = voxelize_hits(hits, requested_voxel_size)
    hits_c, voxels_c = voxelize_hits(hits_c, requested_voxel_size, energy_type)

    # The first assertion is needed for the test to keep being meaningful,
    # in case we change the default value of energy_type to energy_c.
    # problem is, we no longer retain energy type in voxels
    assert not np.isclose(voxels.e, voxels_c.e).all()

    assert np.isclose(voxels.e, hits.groupby('voxel_id').E.sum()).all()

    energies_c = voxels_c.e.values
    e_thr                       = min(energies_c) + fraction_zero_one * (max(energies_c) - min(energies_c))

    # copy again before dropping
    hits_cm   = hits_c.copy(deep=True)
    voxels_cm = voxels_c.copy(deep=True)
    d_hits_c, d_voxels_c, d_dropped_c = drop_voxels(hits_cm, voxels_cm, e_thr, requested_voxel_size, energy_type, min_voxels)

    tot_default_energy     = hits_c.E.sum()
    tot_mod_default_energy = hits_cm.E.sum()

    # We don't want to modify the default energy of hits, if the voxels are made with energy_c
    if len(voxels_cm) < len(voxels_c):
        assert tot_default_energy >= tot_mod_default_energy


def test_make_tracks_function(ICDATADIR):

    # Get some test data
    hit_file    = os.path.join(ICDATADIR, 'tracks_0000_6803_trigger2_v0.9.9_20190111_krth1600.h5')
    evt_number  = 19
    size        = 15.
    voxel_size  = np.array([size,size,size], dtype=np.float16)
    blob_radius = 21*units.mm

    # Read the hits and voxelize
    all_hits = pd.read_hdf(hit_file, "/RECO/Events")

    for evt_number, evt_hits in all_hits.groupby("event", as_index=False):
        evt_time = evt_hits.time.iloc[0]
        hits, voxels   = voxelize_hits(evt_hits, voxel_size, energy_type=HitEnergy.E)

        tracks                   = list(make_track_graphs(voxels, voxel_size))

        hits, voxels, track_coll = make_tracks(hits, voxels, voxel_size,
                                               blob_radius=blob_radius,
                                               energy_type=HitEnergy.E)

        tracks.sort          (key=lambda x : len(x.nodes()))
        tracks_from_coll = track_coll.sort_values('numb_of_voxels')

        # Compare the two sets of tracks
        assert len(tracks) == len(tracks_from_coll)
        for i in range(len(tracks)):
            t  = tracks[i]
            tc = tracks_from_coll.iloc[i]

            assert len(t.nodes()) == tc.numb_of_voxels
            assert np.isclose(sum(voxels[voxels.index.isin(t.nodes())].e), tc.energy).all()

            tc_eblob1 = tc.eblob1
            tc_eblob2 = tc.eblob2

            # calculate blob energies
            extreme_low, extreme_high, length = find_extrema_and_length(t, voxels)
            e_1, e_2, _, _, _, _ = blob_energies_hits_and_centres(t, hits, voxels, blob_radius, extreme_low, extreme_high, voxel_size)

            assert np.allclose(e_1, tc_eblob1)
            assert np.allclose(e_2, tc_eblob2)
