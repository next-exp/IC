import os
import numpy  as np
import pandas as pd

from pytest import mark
from pytest import raises
from pytest import fixture

from ..     reco.psf_functions   import hdst_psf_processing
from ..     reco.psf_functions   import add_variable_weighted_mean
from ..     reco.psf_functions   import add_empty_sensors_and_normalize_q
from ..     reco.psf_functions   import create_psf

from .. database                 import load_db
from ..       io.dst_io          import load_dst
from ..     core.testing_utils   import assert_dataframes_close


def test_add_variable_weighted_mean(ICDATADIR):
    PATH_IN = os.path.join(ICDATADIR, "exact_Kr_tracks_with_MC.h5")

    hdst    = load_dst(PATH_IN, 'RECO', 'Events')
    x_mean  = np.average(hdst.loc[:, 'X'], weights=hdst.loc[:, 'E'])
    y_mean  = np.average(hdst.loc[:, 'Y'], weights=hdst.loc[:, 'E'])

    add_variable_weighted_mean(hdst, 'X', 'E', 'Xpeak')
    add_variable_weighted_mean(hdst, 'Y', 'E', 'Ypeak')

    assert np.allclose(x_mean, hdst.Xpeak.unique())
    assert np.allclose(y_mean, hdst.Ypeak.unique())


def test_add_empty_sensors_and_normalize_q_single_sensor():
    """
    A single sensor hit. The peak is also at the same position. Easy to
    calculate the expected sensors depending on the range. Here we select +- 1
    unit of pitch, so in total there should be 9 sensors.
    """
    hdst  = pd.DataFrame(dict(X=[5], Y=[25], Z=0, Xpeak=5, Ypeak=25, Q=1, E=1, Ec=1))
    hdst2 = add_empty_sensors_and_normalize_q( hdst
                                             , ranges=[ [-11, 11], [-11, 11] ]
                                             , database = load_db.DataSiPM("new", 0)
                                             )

    # notice that we are skipping a few columns in the input data, the number of
    # columns in a realistic dataframe will have a few more columns
    assert hdst2.shape == (9, 10)
    assert np.allclose(hdst2.X.unique(), [-5,  5, 15])
    assert np.allclose(hdst2.Y.unique(), [15, 25, 35])
    assert np.isclose(hdst2.E .sum(), 1)
    assert np.isclose(hdst2.Ec.sum(), 1)
    assert np.isclose(hdst2.Q .sum(), 1)


def test_add_empty_sensors_and_normalize_q_two_sensors():
    """
    Same as the test above, but with two sensors.
    """
    hdst  = pd.DataFrame(dict(X=[-5, 5], Y=[-25, 25], Z=0, Xpeak=0, Ypeak=0, Q=1, E=1, Ec=1))
    hdst2 = add_empty_sensors_and_normalize_q( hdst
                                             , ranges=[ [-26, 26], [-26, 26] ]
                                             , database = load_db.DataSiPM("new", 0)
                                             )

    # notice that we are skipping a few columns in the input data, the number of
    # columns in a realistic dataframe will have a few more columns
    assert hdst2.shape == (36, 10)
    assert np.allclose(hdst2.X.unique(), np.arange(-25, 26, 10))
    assert np.allclose(hdst2.Y.unique(), np.arange(-25, 26, 10))
    assert np.isclose(hdst2.E .sum(), 2)
    assert np.isclose(hdst2.Ec.sum(), 2)
    assert np.isclose(hdst2.Q .sum(), 2)


def test_add_empty_sensors_and_normalize_q_conserves_qe(ICDATADIR):
    PATH_IN        = os.path.join(ICDATADIR, "exact_Kr_tracks_with_MC.h5")
    hdst           = load_dst(PATH_IN, 'RECO', 'Events')
    group          = hdst.groupby('event time npeak'.split())
    hdst_processed = group.apply(add_empty_sensors_and_normalize_q    ,
                                 var      = ['X', 'Y']                ,
                                 ranges   = [[-50, 50], [-50, 50]]    ,
                                 database = load_db.DataSiPM('new', 0),
                                 include_groups=False)
    hdst_processed.reset_index(level=3, inplace=True, drop=True )
    hdst_processed.reset_index(         inplace=True, drop=False)

    assert np.allclose(hdst_processed.groupby('event').NormQ.sum().values, 1.0)
    assert np. isclose(hdst_processed.E.sum(), hdst.E.sum())
    assert np. isclose(hdst_processed.Q.sum(), hdst.Q.sum())


@fixture
def hdst_for_psf():
    return pd.DataFrame(dict( event = 0
                            , time  = 0
                            , npeak = 0
                            , X     = [- 5,  5]
                            , Y     = [-25, 25]
                            , Z     = 12.34
                            , Xpeak = 0
                            , Ypeak = 0
                            , E     = 1
                            , Q     = 1
                            , Ec    = 1
                            ))

def test_hdst_psf_processing_single_event(hdst_for_psf):
    """
    Use a dummy event and verify that it works as expected. We only test here
    the added columns, since the part related to
    `add_empty_sensors_and_normalize_q` has already been tested separately.
    """
    hdst2 = hdst_psf_processing( hdst_for_psf
                               , ranges = [[-26, 26], [-26, 26]]
                               , database = load_db.DataSiPM("new", 0))

    assert hdst2.shape == (36, 17)
    assert np. isclose(hdst2.Zpeak.unique(), hdst_for_psf.Z.mean())
    assert np.allclose(hdst2.RelX.unique(), np.arange(-25, 26, 10))
    assert np.allclose(hdst2.RelY.unique(), np.arange(-25, 26, 10))
    assert np.allclose(hdst2.RelZ.unique(), [0.0])


def test_hdst_psf_processing_duplicated_event(hdst_for_psf):
    """
    Verify that when duplicating the event, the results are consistent.
    """
    hdst  = pd.concat([hdst_for_psf, hdst_for_psf.assign(event=1)], ignore_index=True)
    hdst2 = hdst_psf_processing( hdst
                               , ranges = [[-26, 26], [-26, 26]]
                               , database = load_db.DataSiPM("new", 0))

    assert hdst2.shape == (36*2, 17)

    # need to reset indices and drop event column for comparison
    evt0 = hdst2.groupby("event").get_group(0).reset_index(drop=True).drop(columns=["event"])
    evt1 = hdst2.groupby("event").get_group(1).reset_index(drop=True).drop(columns=["event"])
    assert_dataframes_close(evt0, evt1)


def test_create_psf(ICDATADIR):
    PATH_IN        = os.path.join(ICDATADIR, "exact_Kr_tracks_with_MC_psf.h5")
    PATH_TEST      = os.path.join(ICDATADIR, "test_psf.npz")

    hdst           = pd.read_hdf(PATH_IN)
    psf            = np.load(PATH_TEST)

    bin_edges = [np.linspace(-50, 50, 101) for i in range(2)]
    psf_val, entries, binss = create_psf((hdst.RelX, hdst.RelY), hdst.NormQ, bin_edges)

    assert np.allclose(psf['psf'    ], psf_val)
    assert np.allclose(psf['entries'], entries)
    assert np.allclose(psf['bins'   ],   binss)

@mark.parametrize('ndim', (1, 3))
def test_create_psf_fails_param_dim(ICDATADIR, ndim):
    PATH_IN        = os.path.join(ICDATADIR, "exact_Kr_tracks_with_MC_psf.h5")
    hdst           = pd.read_hdf(PATH_IN)
    bin_edges      = [np.linspace(-50, 50, 101) for i in range(ndim)]

    with raises(ValueError):
        psf_val, entries, binss = create_psf((hdst.RelX, hdst.RelY), hdst.NormQ, bin_edges)


def test_create_psf_fails_ndim(ICDATADIR):
    PATH_IN        = os.path.join(ICDATADIR, "exact_Kr_tracks_with_MC_psf.h5")
    hdst           = pd.read_hdf(PATH_IN)
    bin_edges      = [np.linspace(-50, 50, 101) for i in range(3)]

    with raises(NotImplementedError):
        psf_val, entries, binss = create_psf((hdst.RelX, hdst.RelY, hdst.RelY), hdst.NormQ, bin_edges)
