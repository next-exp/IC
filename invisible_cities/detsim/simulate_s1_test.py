import os
import numpy as np

from .. io.dst_io  import load_dst

from . simulate_s1 import compute_scintillation_photons
from . simulate_s1 import compute_s1_pes_at_pmts

from . light_tables import create_lighttable_function

from pytest import fixture
from pytest import raises

from hypothesis import given, settings
from hypothesis.strategies  import floats
from hypothesis.extra.numpy import arrays


@fixture(scope='session')
def lighttable_filenames(ICDATADIR):
    s1ltfname = os.path.join(ICDATADIR, 'NEXT_NEW.energy.S1.PmtR11410.LightTable.h5')
    s2ltfname = os.path.join(ICDATADIR, 'NEXT_NEW.energy.S2.PmtR11410.LightTable.h5')
    return  {'s1': s1ltfname,
             's2': s2ltfname}


def test_compute_scintillation_photons():
    np.random.seed(1234)
    energy = np.array([1, 2, 3, 4, 5])
    ws = 1.
    expected = np.array([0, 4, 5, 5, 5])
    np.testing.assert_allclose(expected, compute_scintillation_photons(energy, ws))


def test_compute_s1_pes_at_pmts(lighttable_filenames):
    s1_lt = create_lighttable_function(lighttable_filenames["s1"])

    xs = np.array([1, 2]); ys = np.array([1, 2]); zs = np.array([1, 2])
    photons = np.array([1000, 1000])
    s1_pes_at_pmts = compute_s1_pes_at_pmts(xs, ys, zs, photons, s1_lt)

    assert s1_pes_at_pmts.shape == (12, len(xs))
    
