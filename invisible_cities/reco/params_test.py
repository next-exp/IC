import numpy as np

import invisible_cities.core.fit_functions as     fitf
from   invisible_cities.reco.params        import Correction

from numpy.testing import assert_equal
from pytest import fixture, mark


@fixture
def gauss_data_1d():
    mean = lambda z: 1e4 * np.exp(-z/300)
    Nevt = 100000
    zevt = np.random.uniform(0, 500, size=Nevt)
    Eevt = np.random.normal(mean(zevt), mean(zevt)**0.5)
    prof = fitf.profileX(zevt, Eevt, 50, (0, 500))
    return zevt, Eevt, prof


@fixture
def gauss_data_2d():
    mean = lambda x, y: 1e4 * np.exp(-(x**2 + y**2)/400**2)
    Nevt = 100000
    xevt = np.random.uniform(-200, 200, size=Nevt)
    yevt = np.random.uniform(-200, 200, size=Nevt)
    Eevt = np.random.normal(mean(xevt, yevt), mean(xevt, yevt)**0.5)
    prof = fitf.profileXY(xevt, yevt, Eevt, 50, 50, (-200, 200), (-200, 200))
    return xevt, yevt, Eevt, prof


def test_correction_normalize_max(gauss_data_1d):
    x, y, ye = gauss_data_1d[-1]
    c = Correction(x, y, ye, "max")
    assert np.min(c.fs) == 1
    assert np.argmin(c.fs) == 0


def test_correction_normalize_center(gauss_data_2d):
    x, y, z, ze = gauss_data_2d[-1]
    c = Correction((x, y), z, ze, "center")
    assert_equal(np.argwhere(c.fs==1)[0], np.array(z.shape)//2)


def test_correction_unnormalized():
    c = Correction(np.linspace(0, 50, 100), np.arange(100), np.ones(100))
    assert_equal(c.fs, np.arange(100))


@mark.slow
def test_corrections_1d(gauss_data_1d):
    zevt, Eevt, (z, E, Ee) = gauss_data_1d
    zcorr = Correction(z, E, Ee, "max")
    Eevt *= zcorr(zevt)[0]

    y, x = np.histogram(Eevt, np.linspace(7e3, 13e3, 100))
    x = x[:-1] + np.diff(x) * 0.5
    f = fitf.fit(fitf.gauss, x, y, (1e5, 1e4, 1e2))
    assert f.chi2 < 3


@mark.slow
def test_corrections_2d(gauss_data_2d):
    xevt, yevt, Eevt, (x, y, E, Ee) = gauss_data_2d
    xycorr = Correction((x, y), E, Ee, "center")
    Eevt  *= xycorr(xevt, yevt)[0]

    y, x = np.histogram(Eevt, np.linspace(7e3, 13e3, 100))
    x = x[:-1] + np.diff(x) * 0.5
    f = fitf.fit(fitf.gauss, x, y, (1e5, 1e4, 1e2))
    assert f.chi2 < 3