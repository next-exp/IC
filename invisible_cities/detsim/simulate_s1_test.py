import os
import numpy as np

from .. io.dst_io  import load_dst

from . simulate_s1 import compute_scintillation_photons
from . simulate_s1 import compute_s1_pes_at_pmts
from . simulate_s1 import generate_s1_time
from . simulate_s1 import generate_s1_times_from_pes

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


def test_generate_s1_time():
    np.random.seed(1234)
    times = generate_s1_time(size=2)
    expected = np.array([ 57.57690962, 153.87864745])

    np.testing.assert_allclose(times, expected)


def test_generate_s1_times_from_pes():

    np.random.seed(1234)
    s1_pes_at_pmts = np.array([[1, 0, 1],
                               [0, 2, 1]])  # 2 pmts, 3 hits
    hit_times = np.array([0, 1, 2])

    expected  = [np.array([ 57.57690962, 155.87864745]),
                 np.array([ 18.29440868,  13.34541421, 170.82805666])]
    values = generate_s1_times_from_pes(s1_pes_at_pmts, hit_times)

    assert len(values) == len(expected)
    for exp, val in zip(expected, values):
        np.testing.assert_allclose(exp, val)


@settings(max_examples=500)
@given(hit_times = arrays(np.float, 10, elements=floats(min_value=0)))
def test_generate_s1_times_from_pes_higher_times(hit_times):

    s1_pes_at_pmts = np.ones((1, 10), dtype=np.int) # just 1 pmt
    s1_times = generate_s1_times_from_pes(s1_pes_at_pmts, hit_times)[0]

    assert np.all(s1_times>=hit_times)
