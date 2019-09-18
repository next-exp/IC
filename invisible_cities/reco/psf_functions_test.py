import os
import numpy  as np
import pandas as pd

from .. reco.psf_functions   import hdst_PSF_processing
from .. reco.psf_functions   import createPSF

from ..   io.dst_io          import load_dst


def test_hdst_PSF_processing(ICDATADIR):
    PATH_IN        = os.path.join(ICDATADIR, "exact_Kr_tracks_with_MC.h5")
    PATH_TEST      = os.path.join(ICDATADIR, "exact_Kr_tracks_with_MC_psf.h5")

    hdst           = load_dst(PATH_IN, 'RECO', 'Events')
    hdst_psf       = pd.read_hdf(PATH_TEST)

    hdst_processed = hdst_PSF_processing(hdst, [[-50, 50], [-50, 50]], [10, 10])

    assert hdst_psf.equals(hdst_processed)


def test_createPSF(ICDATADIR):
    PATH_IN        = os.path.join(ICDATADIR, "exact_Kr_tracks_with_MC_psf.h5")
    PATH_TEST      = os.path.join(ICDATADIR, "test_psf.npz")

    hdst           = pd.read_hdf(PATH_IN)
    psf            = np.load(PATH_TEST)

    psf_val, entries, binss = createPSF((hdst.RelX, hdst.RelY), hdst.NormQ, [100, 100], [[-50, 50], [-50, 50]])

    np.testing.assert_allclose(psf['psf'    ], psf_val)
    np.testing.assert_allclose(psf['entries'], entries)
    np.testing.assert_allclose(psf['bins'   ],   binss)
