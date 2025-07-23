import numpy  as np
import pandas as pd

from functools import partial

from pytest import fixture
from pytest import mark
from pytest import raises
parametrize = mark.parametrize

from hypothesis             import given
from hypothesis             import settings
from hypothesis.strategies  import floats
from hypothesis.strategies  import integers
from hypothesis.strategies  import composite
from hypothesis.extra.numpy import arrays

from .. core.testing_utils   import assert_cluster_equality
from .. core.testing_utils   import float_arrays
from .. database.load_db     import DataSiPM
from .. core                 import system_of_units as units
from .. core.exceptions      import SipmEmptyList
from .. core.exceptions      import ClusterEmptyList
from .. core.exceptions      import SipmEmptyListAboveQthr
from .. core.exceptions      import SipmZeroCharge

from .       xy_algorithms   import corona
from .       xy_algorithms   import barycenter
from .       xy_algorithms   import discard_sipms
from .       xy_algorithms   import get_nearby_sipm_inds
from .       xy_algorithms   import count_masked


@composite
def positions_and_qs(draw, min_value=1, max_value=100):
    size = draw(integers(min_value, max_value))
    pos  = draw(arrays(float, (size, 2), elements = floats(0.1,1)))
    qs   = draw(arrays(float, (size,  ), elements = floats(0.1,1)))
    return pos, qs


@fixture(scope="session")
def toy_sipm_signal():
    xs  = np.array([65, -55]) * units.mm
    ys  = np.array([65, -75]) * units.mm
    qs  = np.array([ 6,   5]) * units.pes
    pos = np.stack((xs, ys), axis=1)
    return pos, qs


def _create_fake_datasipm(x, y, active):
    assert len(x) == len(y) == len(active)
    size = len(x)
    sensorid   = np.arange(size, dtype=int  )
    channelid  = np.arange(size, dtype=int  )
    adc_to_pes = np.ones  (size, dtype=float)
    sigma      = np.ones  (size, dtype=float)
    active     =         active.astype(int  )
    x          =              x.astype(float)
    y          =              y.astype(float)

    return pd.DataFrame(dict( SensorID  =   sensorid,
                             ChannelID  =  channelid,
                             Active     =     active,
                             X          =          x,
                             Y          =          y,
                             adc_to_pes = adc_to_pes,
                             Sigma      =      sigma))

@fixture(scope="session")
def toy_sipm_signal_and_inds():
    k   = 10000
    xs  = np.arange(k)
    ys  = xs + k
    pos = np.stack((xs, ys), axis=1)
    qs  = xs + 2*k
    i   = np.array([0, 5, 1000, 9999])
    return k, i, pos, qs


@fixture(scope="session")
def datasipm():
    return DataSiPM("new", 0)


@fixture(scope="session")
def datasipm_all_active():
    return DataSiPM("new", 1)


@fixture(scope="session")
def datasipm_5000():
    return DataSiPM("new", 5000)


@fixture(scope="session")
def datasipm_3x5():
    # Create fake database with this 3 x 5 grid of SiPMs:
    # o = active, x = masked
    #
    # x - - - >
    # y | o o o
    #   | o o x
    #   | o o o
    #   | o o o
    #   v o o o
    x         = np.tile  (np.arange(3), 5)
    y         = np.repeat(np.arange(5), 3)
    active    = np.ones  (15, dtype=bool)
    active[5] = False
    return _create_fake_datasipm(x, y, active)


@fixture(scope="session")
def datasipm5x5():
    # Create fake database with this 5 x 5 grid of SiPMs:
    # o = active, x = masked
    #
    # o o o o o
    # o x x x o
    # o x o x o
    # o x x x o
    # o o o o o
    active           = np.ones  ((5, 5), dtype=bool)
    active[1:4, 1:4] = False
    active[  2,   2] = True
    x                = np.tile  (np.arange(5), 5)
    y                = np.repeat(np.arange(5), 5)
    return _create_fake_datasipm(x, y, active.flatten())


@given(positions_and_qs())
@settings(max_examples=100)
def test_barycenter_generic(p_q):
    pos, qs = p_q
    B  = barycenter(pos, qs)[0]
    assert np.allclose(B.posxy, np.average(pos, weights=qs, axis=0))
    assert np. isclose(B.Q  , qs.sum())
    assert B.nsipm == len(qs)


@parametrize("     x         y        q     expected_xy".split(),
             (( (-1, +1), ( 0,  0), (1, 1),   ( 0, 0)  ),
              ( ( 0,  0), (-1, +1), (1, 1),   ( 0, 0)  ),
              ( (-1, +1), (-1, +1), (1, 1),   ( 0, 0)  ),
              ( (-4, +3), (+2, -5), (2, 5),   ( 1,-3)  )))
def test_barycenter_simple_cases(x, y, q, expected_xy):
    xy = np.stack((x, y), axis=1)
    qs = np.array(q)
    b  = barycenter(xy, qs)[0]
    assert np.allclose(b.posxy, expected_xy)


@given(float_arrays(size=8, min_value=0, max_value=100, mask=np.sum))
@settings(max_examples=10)
def test_barycenter_single_cluster_concrete_case(q):
    # This is a double cluster with 4 sipms separated by ~200 mm
    x  = [-105, -105, -95, -95, 95, 95, 105, 105]
    y  = [- 55, - 65, -55, -65, 15, 15,   5,   5]

    xy = np.stack((x, y), axis=1)
    qs = np.array(q)

    clusters = barycenter(xy, qs)
    assert len(clusters) == 1


@given(positions_and_qs())
@settings(max_examples=100)
def test_barycenter_single_cluster_generic(p_q):
    xy, qs   = p_q
    clusters = barycenter(xy, qs)
    assert len(clusters) == 1


corona_default = partial( corona
                        , all_sipms = DataSiPM("new", 0)
                        , Qthr = 0., Qlm = 0.
                        , lm_radius = 0., new_lm_radius = 0.
                        , msipm = 0, consider_masked = False)

@parametrize("algorithm", (barycenter, corona_default))
def test_raises_sipm_empty_list(algorithm):
    with raises(SipmEmptyList):
        algorithm(np.array([]), np.array([]))


@parametrize("algorithm", (barycenter, corona_default))
def test_raises_sipm_zero_charge(algorithm):
    with raises(SipmZeroCharge):
        algorithm(np.array([[1, 2], [3, 4]]), np.array([0, 0]))


def test_corona_converges_to_barycenter(toy_sipm_signal, datasipm):
    """
    Demonstrate that in the case where `new_lm_radius` is infinite,
    corona converges to a simple `barycenter`.
    """
    pos, qs = toy_sipm_signal
    c_clusters = corona(pos, qs, datasipm,
                            lm_radius =  0,
                        new_lm_radius = 10 * units.m,
                        msipm         =  1,
                        Qlm           =  4.9 * units.pes,
                        Qthr          =  0)
    b_clusters = barycenter(pos, qs)

    assert len(c_clusters) == len(b_clusters) == 1
    assert_cluster_equality(c_clusters[0], b_clusters[0])


def test_corona_multiple_clusters(toy_sipm_signal, datasipm):
    """notice: cluster.XY    = (x,y)
               cluster.posxy = ([x],
                                [y])
    """
    pos, qs = toy_sipm_signal
    clusters = corona(pos, qs, datasipm,
                      Qthr = 0, Qlm=4.9*units.pes,
                      lm_radius = 0, new_lm_radius=15*units.mm,
                      msipm = 1)
    assert len(clusters) == 2
    for i in range(len(pos)):
        assert np.array_equal(clusters[i].XY, pos[i])
        assert clusters[i].Q == qs[i]


def test_corona_min_threshold_Qthr(datasipm):
    """notice: cluster.XY =(x,y)
               cluster.posxy = ([x],
                              [y])
    """
    xs = np.arange(100) * 10
    ys = np.zeros (100)
    qs = np.arange(100)
    pos = np.stack((xs, ys), axis=1)

    clusters = corona(pos, qs, datasipm,
                      Qthr           = 99 * units.pes,
                      Qlm            =  1 * units.pes,
                      lm_radius      =  1 * units.mm,
                      new_lm_radius  =  2 * units.mm,
                      msipm          =  1)

    assert len(clusters) ==   1
    assert clusters[0].Q ==  99
    assert clusters[0].XY == (990, 0)


def test_corona_msipm(toy_sipm_signal, datasipm):
    pos, qs = toy_sipm_signal
    with raises(ClusterEmptyList):
        corona( pos, qs, datasipm
              , Qthr=0, Qlm=5 * units.pes
              , lm_radius=0, new_lm_radius=15 * units.mm
              , msipm=2)


@parametrize(' Qlm,    rmax, nclusters',
             ((4.9,      15,         2),
              (4.9, 1000000,         1)))
def test_corona_simple_examples(toy_sipm_signal, datasipm, Qlm, rmax, nclusters):
    pos, qs  = toy_sipm_signal
    clusters = corona(pos, qs, datasipm,
                      Qthr            =  0,
                      Qlm             =  Qlm * units.pes,
                      lm_radius       =  0,
                      new_lm_radius   = rmax * units.mm,
                      msipm           =  1,
                      consider_masked = False
                      )
    assert len(clusters) == nclusters


def test_corona_Qlm_too_high_raises_ClusterEmptyList(toy_sipm_signal, datasipm):
    pos, qs  = toy_sipm_signal
    Qlm      = max(qs) * 1.1 * units.pes

    with raises(ClusterEmptyList):
        corona(pos, qs, datasipm,
               Qthr           =      0,
               Qlm            =    Qlm,
                   lm_radius  =      0,
               new_lm_radius  = np.inf,
               msipm          =      1)


def test_corona_Qthr_too_high_raises_SipmEmptyListAboveQthr(toy_sipm_signal, datasipm):
    pos, qs  = toy_sipm_signal
    Qthr     = max(qs) * 1.1 * units.pes

    with raises(SipmEmptyListAboveQthr):
        corona(pos, qs, datasipm,
               Qthr           =          Qthr,
               Qlm            = 5 * units.pes,
               msipm          =             1,
                   lm_radius  =             0,
               new_lm_radius  =        np.inf)


def test_discard_sipms(toy_sipm_signal_and_inds):
    k, i, pos, qs = toy_sipm_signal_and_inds
    xysel, qsel = discard_sipms(i, pos, qs)
    xsel , ysel = xysel.T
    for ind in i:
        assert ind         not in xsel
        assert ind +     k not in ysel
        assert ind + 2 * k not in qsel


def test_get_nearby_sipm_inds():
    xs  = np.array([0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4])
    ys  = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4])
    pos = np.stack((xs, ys), axis=1)
    xc, yc = (2, 2)
    c = np.array((xc, yc))
    d  = 1.5
    sis = get_nearby_sipm_inds(c, d, pos)
    for i in range(len(xs)):
        if i in sis: assert np.sqrt((xs[i] - xc)**2 + (ys[i] - yc)**2) <= d
        else       : assert np.sqrt((xs[i] - xc)**2 + (ys[i] - yc)**2) >  d


def test_count_masked_all_active(datasipm_all_active):
    xy0 = np.array([0, 0], dtype=float)
    is_masked = datasipm_all_active.Active.values

    # All sipms are active in run number 1
    assert count_masked(xy0, np.inf, datasipm_all_active) == 0


@mark.parametrize("sipm_id  radius  expected_nmasked".split(),
                  (( 14010,      5,                1),
                   ( 14010,     15,                1),
                   (  1006,      5,                1),
                   (  1006,     15,                2),
                   ( 26062,      5,                1),
                   ( 26062,     15,                2)))
def test_count_masked_near_masked(datasipm_5000, sipm_id, radius, expected_nmasked):
    sipm_indx    = np.argwhere(datasipm_5000.SensorID.values == sipm_id)[0][0]
    masked_sipm  = datasipm_5000.iloc[sipm_indx]
    masked_xy    = np.array([masked_sipm.X, masked_sipm.Y])

    # small smear so the search point doesn't fall exactly at sipm position
    masked_xy   += np.random.normal(0, 0.001 * radius, size=2)
    assert count_masked(masked_xy, radius, datasipm_5000) == expected_nmasked


def test_masked_channels(datasipm_3x5):
    """
    Scheme of SiPM positions (the numbers are the SiPM charges)
    x - - - >
    y | 1 1 1
      | 1 5 0
      | 1 1 1
      | 1 6 1
      v 1 1 1

    This test is meant to fail if either
    1) in the case of an empty masked channel list,
       the actual threshold in the number of SiPMs
       around the hottest one turns out to be different from msipm
    2) the masked channel is not taken properly into account
       by the code
    """
    datasipm   = datasipm_3x5
    x          = np.nan
    qs         = np.array([[1, 1, 1],
                           [1, 5, x],
                           [1, 1, 1],
                           [1, 6, 1],
                           [1, 1, 1]]).flatten()
    pos        = np.stack((datasipm.X.values, datasipm.Y.values), axis=1)
    ok         = ~np.isnan(qs)

    # Corona should return 1 cluster if the masked sipm is taken into account...
    expected_nclusters = 1
    found_clusters = corona(pos[ok], qs[ok], datasipm,
                            Qthr            = 0   * units.pes,
                            Qlm             = 4   * units.pes,
                                lm_radius   = 0   * units.mm ,
                            new_lm_radius   = 1.5 * units.mm ,
                            msipm           = 6              ,
                            consider_masked = False)

    assert len(found_clusters) == expected_nclusters

    # ... and two when ignored.
    expected_nclusters = 2
    found_clusters = corona(pos[ok], qs[ok], datasipm,
                            Qthr            = 0   * units.pes,
                            Qlm             = 4   * units.pes,
                                lm_radius   = 0   * units.mm ,
                            new_lm_radius   = 1.5 * units.mm ,
                            msipm           = 6              ,
                            consider_masked = True)

    assert len(found_clusters) == expected_nclusters


def test_corona_finds_masked_sipms_correctly(datasipm5x5):
    # With the appropiate parameters, this should yield a single cluster
    # with only 25 - 8 = 17 SiPM.
    datasipm = datasipm5x5

    # For visualization purposes we use o = np.nan and then remove
    # those values
    x       = np.nan
    all_xys = np.stack([datasipm.X.values, datasipm.Y.values], axis=1)
    all_qs  = np.array([[1, 2, 3, 4, 9], # create slight asymmetry
                        [2, x, x, x, 4], # so the barycenter doesn't
                        [3, x, 9, x, 3], # fall exactly at the center
                        [4, x, x, x, 2], # of the array
                        [5, 4, 3, 2, 1]], dtype=float).flatten()

    ok = ~np.isnan(all_qs)

    c = corona(all_xys[ok], all_qs[ok], datasipm,
               Qthr            =      0,
               Qlm             =      6,
                   lm_radius   =      0,
               new_lm_radius   = np.inf, # take all sipms
               msipm           =     25, # request all sipms
               consider_masked =   True)

    assert len(c)     ==  1
    assert c[0].nsipm == 17
