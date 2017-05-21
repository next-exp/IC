import numpy as np

from ..core        import fit_functions as fitf
from ..reco.params import Correction

from numpy.testing import assert_equal, assert_allclose
from pytest        import fixture, mark
from collections   import namedtuple


data_1d = namedtuple("data_1d",   "z E prof")
data_2d = namedtuple("data_2d", "x y E prof")


@fixture(scope='session')
def toy_data_1d():
    nx = 10
    X  = np.linspace      (  0, 100, nx)
    E  = np.random.uniform(1e3, 2e3, nx)
    Eu = np.random.uniform(1e2, 2e2, nx)

    i_max = np.argmax(E)
    e_max = E [i_max]
    u_max = Eu[i_max]

    F  = E.max()/E
    Fu = F * (Eu**2/E**2 + u_max**2/e_max**2)**0.5
    C  = Correction((X,), E, Eu, "max")
    return X, E, Eu, F, Fu, C


@fixture(scope='session')
def toy_data_2d():
    nx = 10
    ny = 20
    X  = np.linspace      (  0, 100,  nx     )
    Y  = np.linspace      (100, 200,      ny )
    E  = np.random.uniform(1e3, 2e3, (nx, ny))
    Eu = np.random.uniform(1e2, 2e2, (nx, ny))

    i_max = nx//2, ny//2
    e_max = E [i_max]
    u_max = Eu[i_max]
    print(i_max, e_max, u_max)
    F  = e_max/E
    Fu = F * (Eu**2/E**2 + u_max**2/e_max**2)**0.5
    C  = Correction((X, Y), E, Eu, "index", index=i_max)
    return X, Y, E, Eu, F, Fu, C


def test_correction_attributes_1d(toy_data_1d):
    X, E, Eu, F, Fu, correct = toy_data_1d
    assert_allclose(correct.xs[0], X ) # correct.xs is a list of axis
    assert_allclose(correct.fs   , F )
    assert_allclose(correct.us   , Fu)


def test_correction_call_1d(toy_data_1d):
    X, E, Eu, F, Fu, correct = toy_data_1d

    X_test = X
    F_test, U_test = correct(X_test)
    assert_allclose(F_test, F )
    assert_allclose(U_test, Fu)

def test_correction_attributes_2d(toy_data_2d):
    X, Y, E, Eu, F, Fu, correct = toy_data_2d
    assert_allclose(correct.fs, F )
    assert_allclose(correct.us, Fu)


def test_correction_call_2d(toy_data_2d):
    X, Y, E, Eu, F, Fu, correct = toy_data_2d

    # Combine x and y so they are paired
    X_test = np.repeat(X, Y.size)
    Y_test = np.tile  (Y, X.size)
    F_test, U_test = correct(X_test, Y_test)
    assert_allclose(F_test, F .flatten())
    assert_allclose(U_test, Fu.flatten())



"""
@fixture(scope='session')
def toy_data_2d():
    x    = np.linspace(  0, 100, 10)
    y    = np.linspace(  0, 100, 10)
    E    = np.linspace(1e3, 2e3, x.size*y.size).reshape(x.size, y.size)
    return data_2d(x, y, E, None)


@fixture(scope='session')
def gauss_data_1d():
    mean = lambda z: 1e4 * np.exp(-z/300)
    Nevt = 100000
    zevt = np.random.uniform(0, 500, size=Nevt)
    Eevt = np.random.normal(mean(zevt), mean(zevt)**0.5)
    prof = fitf.profileX(zevt, Eevt, 50, (0, 500))
    return data_1d(zevt, Eevt, prof)


@fixture(scope='session')
def gauss_data_2d():
    mean = lambda x, y: 1e4 * np.exp(-(x**2 + y**2)/400**2)
    Nevt = 100000
    xevt = np.random.uniform(-200, 200, size=Nevt)
    yevt = np.random.uniform(-200, 200, size=Nevt)
    Eevt = np.random.normal(mean(xevt, yevt), mean(xevt, yevt)**0.5)
    prof = fitf.profileXY(xevt, yevt, Eevt, 50, 50, (-200, 200), (-200, 200))
    return data_2d(xevt, yevt, Eevt, prof)


def test_correction_normalize_max(gauss_data_1d):
    x, y, ye = gauss_data_1d.prof
    c = Correction(x, y, ye, "max")
    assert np.min(c.fs) == 1
    assert np.argmin(c.fs) == 0


def test_correction_normalize_center(gauss_data_2d):
    x, y, z, ze = gauss_data_2d.prof
    c = Correction((x, y), z, ze, "center")
    assert_equal(np.argwhere(c.fs==1)[0], np.array(z.shape)//2)


def test_correction_unnormalized():
    c = Correction(np.linspace(0, 50, 100), np.arange(100), np.ones(100))
    assert_equal(c.fs, np.arange(100))


def test_corrections_toy_1d(toy_data_1d):
    z, E, _ = toy_data_1d
    Emax = np.max(E)
    zcorr = Correction(z, E, np.ones_like(E), "max")
    E *= zcorr(z)[0]
    assert_allclose(E, Emax)


def test_corrections_toy_2d(toy_data_2d):
    x, y, E, _  = toy_data_2d
    xycorr      = Correction((x, y), E, np.ones_like(E), "center")
    y, x        = map(np.ndarray.flatten, np.meshgrid(x, y))
    E           = E.flatten()
    E          *= xycorr(x, y)[0]
    assert_allclose(E, np.mean(E))


@mark.slow
def test_corrections_1d(gauss_data_1d):
    zevt, Eevt, (z, E, Ee) = gauss_data_1d
    zcorr = Correction(z, E, Ee, "max")
    Eevt *= zcorr(zevt)[0]

    mean = np.std(Eevt)
    std  = np.std(Eevt)
    y, x = np.histogram(Eevt, np.linspace(mean - 3 * std, mean + 3 * std, 100))
    x = x[:-1] + np.diff(x) * 0.5
    f = fitf.fit(fitf.gauss, x, y, (1e5, mean, std))
    assert f.chi2 < 3


@mark.slow
def test_corrections_2d(gauss_data_2d):
    xevt, yevt, Eevt, (x, y, E, Ee) = gauss_data_2d
    xycorr = Correction((x, y), E, Ee, "center")
    Eevt  *= xycorr(xevt, yevt)[0]

    mean = np.std(Eevt)
    std  = np.std(Eevt)
    y, x = np.histogram(Eevt, np.linspace(mean - 3 * std, mean + 3 * std, 100))
    x = x[:-1] + np.diff(x) * 0.5
    f = fitf.fit(fitf.gauss, x, y, (1e5, mean, std))
    assert f.chi2 < 3
"""