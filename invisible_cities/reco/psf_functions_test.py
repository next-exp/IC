import os
import numpy  as np
import pandas as pd

from ..     reco.psf_functions   import hdst_psf_processing
from ..     reco.psf_functions   import add_variable_weighted_mean
from ..     reco.psf_functions   import add_empty_sensors_and_normalize_q
from ..     reco.psf_functions   import create_psf

from .. database                 import load_db
from ..       io.dst_io          import load_dst
from ..     core.testing_utils   import assert_dataframes_close


def test_add_variable_weighted_mean(ICDATADIR):
    PATH_IN   = os.path.join(ICDATADIR, "exact_Kr_tracks_with_MC.h5")
    PATH_TEST = os.path.join(ICDATADIR, "exact_Kr_tracks_with_MC_psf_means.h5")

    hdst      = load_dst(PATH_IN, 'RECO', 'Events')
    x_mean    = np.average(hdst.loc[:, 'X'], weights=hdst.loc[:, 'E'])
    y_mean    = np.average(hdst.loc[:, 'Y'], weights=hdst.loc[:, 'E'])

    add_variable_weighted_mean(hdst, 'X', 'E', 'Xpeak')
    add_variable_weighted_mean(hdst, 'Y', 'E', 'Ypeak')

    assert np.allclose(x_mean, hdst.Xpeak.unique())
    assert np.allclose(y_mean, hdst.Ypeak.unique())

    hdst_psf  = pd.read_hdf(PATH_TEST)

    assert_dataframes_close(hdst_psf, hdst)


def test_add_empty_sensors_and_normalize_q(ICDATADIR):
    PATH_IN   = os.path.join(ICDATADIR, "exact_Kr_tracks_with_MC.h5")
    PATH_TEST = os.path.join(ICDATADIR, "exact_Kr_tracks_with_MC_psf_empty_sensors.h5")

    hdst           = load_dst(PATH_IN, 'RECO', 'Events')
    group          = hdst.groupby(['event'], as_index=False)
    sipm_db        = load_db.DataSiPM('new', 0)
    hdst_processed = group.apply(add_empty_sensors_and_normalize_q, ['X', 'Y'], [[-50, 50], [-50, 50]], sipm_db).reset_index(drop=True)
    hdst_psf       = pd.read_hdf(PATH_TEST)

    assert hdst_processed.NormQ.sum() == 1.0 * hdst.event.nunique()
    assert hdst_processed.E.sum()     == hdst.E.sum()
    assert hdst_processed.Q.sum()     == hdst.Q.sum()
    assert_dataframes_close(hdst_psf, hdst_processed)


def test_hdst_psf_processing(ICDATADIR):
    PATH_IN        = os.path.join(ICDATADIR, "exact_Kr_tracks_with_MC.h5")
    PATH_TEST      = os.path.join(ICDATADIR, "exact_Kr_tracks_with_MC_psf.h5")

    hdst           = load_dst(PATH_IN, 'RECO', 'Events')
    hdst_processed = hdst_psf_processing(hdst, [[-50, 50], [-50, 50]], 'new', 0)
    hdst_psf       = pd.read_hdf(PATH_TEST)

    assert_dataframes_close(hdst_psf, hdst_processed)


def test_create_psf(ICDATADIR):
    PATH_IN        = os.path.join(ICDATADIR, "exact_Kr_tracks_with_MC_psf.h5")
    PATH_TEST      = os.path.join(ICDATADIR, "test_psf.npz")

    hdst           = pd.read_hdf(PATH_IN)
    psf            = np.load(PATH_TEST)

    bin_edges = [np.linspace(-50, 50, 101) for i in range(2)]
    psf_val, entries, binss = create_psf((hdst.RelX, hdst.RelY), hdst.NormQ, bin_edges)

    np.testing.assert_allclose(psf['psf'    ], psf_val)
    np.testing.assert_allclose(psf['entries'], entries)
    np.testing.assert_allclose(psf['bins'   ],   binss)
