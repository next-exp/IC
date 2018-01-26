from collections import namedtuple

import numpy as np

from pytest                import raises
from pytest                import approx
from .. core.testing_utils import exactly
from .. core.testing_utils import assert_xyz_equality
from .. core.testing_utils import assert_cluster_equality
from .. core.testing_utils import assert_hit_equality
from .. core.testing_utils import assert_space_element_equality
from .. core.testing_utils import assert_blob_equality
from .. core.testing_utils import assert_track_equality

from hypothesis            import given
from hypothesis            import settings
from hypothesis            import HealthCheck
from hypothesis.strategies import composite
from hypothesis.strategies import integers
from hypothesis.strategies import floats
from hypothesis.strategies import one_of
from hypothesis.strategies import sampled_from
from .. core.testing_utils import random_length_float_arrays

from . import new_event_model as evm


numbers                   = one_of(integers(-100, 100), floats(-100, 100))
positive_numbers          = one_of(integers(   0, 100), floats(   0, 100))
array_of_positive_numbers = random_length_float_arrays(min_length = 1, max_length =   5,
                                                       min_value  = 0, max_value  = 100)
number_of_things          = integers(1, 5)
space_element_subtypes    = sampled_from((evm.    SpaceElement,
                                          evm.           Voxel,
                                          evm.      CubicVoxel,
                                          evm.EllipsoidalVoxel,
                                          evm.  SphericalVoxel))
blob_subtypes             = sampled_from((evm.         Blob,
                                          evm.SphericalBlob,
                                          evm.    CubicBlob))


ObjectWithArgs = namedtuple("ObjectWithArgs", "args obj")
obj_with_args  = lambda args, obj: ObjectWithArgs(args, obj(*args))


@composite
def xys(draw):
    x    = draw(numbers)
    y    = draw(numbers)
    args = x, y
    return obj_with_args(args, evm.xy)


@composite
def xyzs(draw):
    x, y = draw(xys()).args
    z    = draw(numbers)
    args = x, y, z
    return obj_with_args(args, evm.xyz)


@composite
def xyzEs(draw):
    x, y, z = draw(xyzs()).args
    E       = draw(positive_numbers)
    args    = x, y, z, E
    return obj_with_args(args, evm.xyzE)


@composite
def mchits(draw):
    x, y, z, E = draw(xyzEs()).args
    t          = draw(positive_numbers)
    args       = x, y, z, E, t
    return obj_with_args(args, evm.MCHit)


@composite
def space_elements(draw):
    x, y, z  = draw(xyzs()).args
    size     = draw(positive_numbers)
    hit_enes = draw(array_of_positive_numbers)
    ene_sum  = np.sum(hit_enes)
    subtype  = draw(space_element_subtypes)
    args     = x, y, z, ene_sum, size, hit_enes
    return obj_with_args(args, subtype)


@composite
def clusters(draw):
    Q      = draw(positive_numbers)
    x, y   = draw(xys()).args
    x_var  = draw(positive_numbers)
    y_var  = draw(positive_numbers)
    n_sipm = draw(number_of_things)
    args   = Q, x, y, x_var, y_var, n_sipm
    return obj_with_args(args, evm.Cluster)


@composite
def hits(draw):
    *_, z, E = draw(xyzEs   ()).args
    cluster  = draw(clusters()).obj
    peak_no  = draw(number_of_things)
    args     = cluster, z, E, peak_no
    return obj_with_args(args, evm.Hit)


@composite
def voxel_collections(draw):
    n_voxels = draw(number_of_things)
    # Do we want to impose that all elements must be of the same (sub)type?
    voxels   = [draw(space_elements()).obj for _ in range(n_voxels)]
    args     = voxels,
    return obj_with_args(args, evm.VoxelCollection)


@composite
def blobs(draw):
    voxels, = draw(voxel_collections()).args
    seed    = draw(xyzs()).obj
    size    = draw(numbers)
    subtype = draw(blob_subtypes)
    args    = voxels, seed, size
    return obj_with_args(args, subtype)


@composite
def tracks(draw):
    voxels,   = draw(voxel_collections()).args
    n_blobs   = draw(number_of_things)
    n_extrema = draw(number_of_things)
    blobs_    = [draw(blobs()).obj for _ in range(n_blobs  )]
    extrema   = [draw( xyzs()).obj for _ in range(n_extrema)]
    args      = voxels, blobs_, extrema
    return obj_with_args(args, evm.Track)


@composite
def events(draw):
    event_number = draw(number_of_things)
    timestamp    = draw(positive_numbers)
    args         = event_number, timestamp
    return obj_with_args(args, evm.Event)


@composite
def hit_collections(draw):
    event_number, timestamp = draw(events()).args
    n_hits                  = draw(number_of_things)
    hits_                   = [draw(hits()).obj for _ in range(n_hits)]
    args                    = event_number, timestamp, hits_
    return obj_with_args(args, evm.HitCollection)


@composite
def track_collections(draw):
    event_number, timestamp = draw(events()).args
    n_tracks                = draw(number_of_things)
    tracks_                 = [draw(tracks()).obj for _ in range(n_tracks)]
    args                    = event_number, timestamp, tracks_
    return obj_with_args(args, evm.TrackCollection)


@given(xys())
def test_xy_attributes(xy_data):
    (x, y), xy = xy_data
    assert xy.x   == approx(x)
    assert xy.y   == approx(y)
    assert xy.xy  == approx((x, y))
    assert xy.r   == approx((x**2 + y**2)**0.5)
    assert xy.phi == approx(np.arctan2(y, x))


@given(xyzs())
def test_xyz_attributes(xyz_data):
    (x, y, z), xyz = xyz_data
    assert xyz.x   == approx(x)
    assert xyz.y   == approx(y)
    assert xyz.z   == approx(z)
    assert xyz.xy  == approx((x, y))
    assert xyz.xyz == approx((x, y, z))
    assert xyz.r   == approx((x**2 + y**2)**0.5)
    assert xyz.phi == approx(np.arctan2(y, x))


@given(xyzEs())
def test_xyz_attributes(xyzE_data):
    (x, y, z, E), xyzE = xyzE_data
    assert xyzE.x    == approx(x)
    assert xyzE.y    == approx(y)
    assert xyzE.z    == approx(z)
    assert xyzE.E    == approx(E)
    assert xyzE.xy   == approx((x, y))
    assert xyzE.xyz  == approx((x, y, z))
    assert xyzE.xyze == approx((x, y, z, E))
    assert xyzE.r    == approx((x**2 + y**2)**0.5)
    assert xyzE.phi  == approx(np.arctan2(y, x))


@given(mchits())
def test_MCHit_attributes(mchit_data):
    (x, y, z, E, t), mchit = mchit_data
    assert mchit.x     == approx(x)
    assert mchit.y     == approx(y)
    assert mchit.z     == approx(z)
    assert mchit.E     == approx(E)
    assert mchit.t     == approx(t)
    assert mchit.xy    == approx((x, y))
    assert mchit.xyz   == approx((x, y, z))
    assert mchit.xyze  == approx((x, y, z, E))
    assert mchit.xyzt  == approx((x, y, z, t))
    assert mchit.xyzet == approx((x, y, z, E, t))
    assert mchit.r     == approx((x**2 + y**2)**0.5)
    assert mchit.phi   == approx(np.arctan2(y, x))


@given(space_elements())
def test_SpaceElement_attributes(space_element_data):
    (x, y, z, E, size, enes), spel = space_element_data
    assert spel.x      == approx (x)
    assert spel.y      == approx (y)
    assert spel.z      == approx (z)
    assert spel.E      == approx (E)
    assert spel.xy     == approx ((x, y))
    assert spel.xyz    == approx ((x, y, z))
    assert spel.xyze   == approx ((x, y, z, E))
    assert spel.r      == approx ((x**2 + y**2)**0.5)
    assert spel.phi    == approx (np.arctan2(y, x))
    assert spel.size   == exactly(size)
    assert spel.hit_es == approx (enes)


@given(space_elements())
def test_SpaceElement_wrong_input_raises_ValueError(space_element_data):
    (x, y, z, E, size, enes), spel = space_element_data
    subtype = type(spel)
    wrong_E = E + 1
    with raises(ValueError):
        subtype(x, y, z, wrong_E, size, enes)


@given(clusters())
def test_Cluster_attributes(cluster_data):
    (Q, x, y, x_var, y_var, n_sipm), cluster = cluster_data
    assert cluster.Q      == approx (Q)
    assert cluster.x      == approx (x)
    assert cluster.y      == approx (y)
    assert cluster.x_var  == approx (x_var)
    assert cluster.y_var  == approx (y_var)
    assert cluster.n_sipm == exactly(n_sipm)


@given(hits())
def test_Hit_attributes(hit_data):
    (cluster, z, E, peak_no), hit = hit_data
    assert hit.x       == approx ( cluster.x)
    assert hit.y       == approx ( cluster.y)
    assert hit.z       == approx (         z)
    assert hit.E       == approx (         E)
    assert hit.xy      == approx ((cluster.x    , cluster.y         ))
    assert hit.xyz     == approx ((cluster.x    , cluster.y   , z   ))
    assert hit.xyze    == approx ((cluster.x    , cluster.y   , z, E))
    assert hit.r       == approx ((cluster.x**2 + cluster.y**2)**0.5)
    assert hit.phi     == approx (np.arctan2(cluster.y, cluster.x))
    assert hit.peak_no == exactly(peak_no)
    assert_cluster_equality(hit.cluster, cluster)


@given(voxel_collections())
def test_VoxelCollection_attributes(voxel_collection_data):
    (voxels,), voxel_collection = voxel_collection_data

    assert len(voxels) == len(voxel_collection.voxels)
    for got, expected in zip(voxel_collection.voxels, voxels):
        assert_space_element_equality(got, expected)

    assert voxel_collection.E == approx(sum(voxel.E for voxel in voxels))


@given(blobs())
def test_Blob_attributes(blob_data):
    (voxels, seed, size), blob = blob_data

    assert len(voxels) == len(blob.voxels)
    for got, expected in zip(blob.voxels, voxels):
        assert_space_element_equality(got, expected)

    assert blob.E      == approx (sum(voxel.E for voxel in voxels))
    assert blob.seed.x == approx (seed.x)
    assert blob.seed.y == approx (seed.y)
    assert blob.seed.z == approx (seed.z)
    assert blob.size   == exactly(size)


@given(tracks())
def test_Track_attributes(track_data):
    (voxels, blobs, extrema), track = track_data
    smallest_blob_index = np.argmin([blob.E for blob in blobs])
    biggest_blob_index  = np.argmax([blob.E for blob in blobs])

    assert len(voxels) == len(track.voxels)
    for got, expected in zip(track.voxels, voxels):
        assert_space_element_equality(got, expected)

    assert track.E == approx(sum(voxel.E for voxel in voxels))

    assert len(blobs) == len(track.blobs)
    for got_blob, expected_blob in zip(track.blobs, blobs):
        assert_blob_equality(got_blob, expected_blob)

    assert len(extrema) == len(track.extrema)
    for got, expected in zip(track.extrema, extrema):
        assert_xyz_equality(got, expected)

    assert_blob_equality(track.smallest_blob, blobs[smallest_blob_index])
    assert_blob_equality(track. biggest_blob, blobs[ biggest_blob_index])

@given(events())
def test_Event_attributes(event_data):
    (event_number, timestamp), event = event_data

    assert event.event_number == exactly(event_number)
    assert event.timestamp    == approx (timestamp)


@given(hit_collections())
def test_HitCollection_attributes(hit_collection_data):
    (event_number, timestamp, hits), hit_collection = hit_collection_data

    assert hit_collection.event_number == exactly(event_number)
    assert hit_collection.timestamp    == approx (timestamp)

    assert len(hit_collection.hits) == len(hits)
    for got, expected in zip(hit_collection.hits, hits):
        assert_hit_equality(got, expected)


# This test takes too long to generate data and fails,
# but this failure has nothing to do with the test itself.
@settings(suppress_health_check=[HealthCheck.too_slow])
@given(track_collections())
def test_TrackCollection_attributes(track_collection_data):
    (event_number, timestamp, tracks), track_collection = track_collection_data

    assert track_collection.event_number == exactly(event_number)
    assert track_collection.timestamp    == approx (timestamp)

    assert len(track_collection.tracks) == len(tracks)
    for got, expected in zip(track_collection.tracks, tracks):
        assert_track_equality(got, expected)
