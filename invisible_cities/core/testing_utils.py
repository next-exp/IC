import numpy  as np

from pytest import approx

from numpy.testing          import assert_array_equal, assert_allclose
from hypothesis.strategies  import integers, floats
from hypothesis.extra.numpy import arrays


def exactly(value, **kwargs):
    """
    Assert that two numbers (or two sets of numbers) are equal to each
    other with zero tolerance. This function is intended to have a
    similar behavior to `pytest`'s approx function, but for integers.
    This is particularly interesting when comparing numpy arrays, as
    the expression is less readable:

    ```
    np.all(array1 == array2)
    ```

    to be compared with

    ```
    array1 == exactly(array2)
    ```

    which is more pleasant to the human eye and keeps the symmetry with
    floating point comparisons.
    """
    return approx(value, rel=0, abs=0, **kwargs)


def previous_float(x):
    """
    Return the next float towards -inf.
    """
    return np.nextafter(x, -np.inf)


def next_float(x):
    """
    Return the next float towards +inf.
    """
    return np.nextafter(x, +np.inf)


def float_arrays(size       =   100,
                 min_value  = -1e20,
                 max_value  = +1e20,
                 mask       =  None,
                 **kwargs          ):
    elements = floats(min_value,
                      max_value,
                      **kwargs)
    if mask is not None:
        elements = elements.filter(mask)
    return arrays(dtype    = np.float32,
                  shape    =       size,
                  elements =   elements)


def FLOAT_ARRAY(*args, **kwargs):
    return float_arrays(*args, **kwargs).example()


def random_length_float_arrays(min_length =     0,
                               max_length =   100,
                               **kwargs          ):
    lengths = integers(min_length,
                       max_length)

    return lengths.flatmap(lambda n: float_arrays(       n,
                                                  **kwargs))


def _compare_dataframes(assertion, df1, df2, check_types=True, **kwargs):
    assert sorted(df1.columns) == sorted(df2.columns), "DataFrames with different structure cannot be compared"

    for col in df1.columns:
        col1 = df1[col]
        col2 = df2[col]
        if check_types:
            assert col1.dtype == col2.dtype
        print(col, col1.values - col2.values)
        assertion(col1.values, col2.values, **kwargs)


def assert_dataframes_equal(df1, df2, check_types=True, **kwargs):
    _compare_dataframes(assert_array_equal, df1, df2, check_types, **kwargs)


def assert_dataframes_close(df1, df2, check_types=True, **kwargs):
    _compare_dataframes(assert_allclose, df1, df2, check_types, **kwargs)


def assert_SensorResponses_equality(sr0, sr1):
    # This is sufficient to assert equality since all of SensorResponses other
    # properties depend solely on .all_waveforms, and all of those properties
    # are tested.
    assert sr0.ids           == exactly(sr1.ids)
    assert sr0.all_waveforms == approx(sr1.all_waveforms)


def assert_Peak_equality(pk0, pk1):
    assert pk0.times == approx(pk1.times)
    assert_SensorResponses_equality(pk0.pmts , pk1.pmts )
    assert_SensorResponses_equality(pk0.sipms, pk1.sipms)


def assert_PMap_equality(pmp0, pmp1):
    assert len(pmp0.s1s) == len(pmp1.s1s)
    assert len(pmp0.s2s) == len(pmp1.s2s)

    for s1_0, s1_1 in zip(pmp0.s1s, pmp1.s1s):
        assert_Peak_equality(s1_0, s1_1)

    for s2_0, s2_1 in zip(pmp0.s2s, pmp1.s2s):
        assert_Peak_equality(s2_0, s2_1)


def assert_xy_equality(xy1, xy2):
    assert xy1.x == approx(xy2.x)
    assert xy1.y == approx(xy2.y)


def assert_xyz_equality(xyz1, xyz2):
    assert_xy_equality(xyz1, xyz2)
    assert xyz1.z == approx(xyz2.z)


def assert_xyzE_equality(xyze1, xyze2):
    assert_xyz_equality(xyze1, xyze2)
    assert xyze1.E == approx(xyze2.E)


def assert_cluster_equality(cluster1, cluster2):
    assert cluster1.Q      == approx (cluster2.Q     )
    assert cluster1.x      == approx (cluster2.x     )
    assert cluster1.y      == approx (cluster2.y     )
    assert cluster1.x_var  == approx (cluster2.x_var )
    assert cluster1.y_var  == approx (cluster2.y_var )
    assert cluster1.n_sipm == exactly(cluster2.n_sipm)


def assert_hit_equality(hit1, hit2):
    assert_xyzE_equality   (hit1, hit2)
    assert_cluster_equality(hit1, hit2)
    assert hit1.peak_no == exactly(hit2.peak_no)


def assert_space_element_equality(se1, se2):
    assert_xyzE_equality(se1, se2)
    assert se1.size   == approx(se2.size  )
    assert se1.hit_es == approx(se2.hit_es)


def assert_voxel_collection_equality(track1, track2):
    assert len(track1.voxels) == len(track2.voxels)
    for v1, v2 in zip(track1.voxels, track2.voxels):
        assert_space_element_equality(v1, v2)


def assert_blob_equality(blob1, blob2):
    assert_voxel_collection_equality(blob1, blob2)
    assert_xyz_equality(blob1.seed, blob2.seed)
    assert blob1.size == approx(blob2.size)


def assert_track_equality(track1, track2):
    assert_voxel_collection_equality(track1, track2)

    assert len(track1.blobs) == len(track2.blobs)
    for b1, b2 in zip(track1.blobs, track2.blobs):
        assert_blob_equality(b1, b2)

    assert len(track1.extrema) == len(track2.extrema)
    for e1, e2 in zip(track1.extrema, track2.extrema):
        assert_xyz_equality(e1, e2)
