import numpy as np

from ..core             import fit_functions as fitf
from ..reco.corrections import Correction, Fcorrection

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


@fixture(scope='session')
def toy_f_data():
    fun   = lambda x, LT: fitf.expo(x, 1, -LT)
    u_fun = lambda u, LT: u/LT**2 * fun(u, LT)
    LT    = 100
    Z     = np.linspace(0, 600, 100)
    u_Z   = np.random.uniform(0.1, 0.2, 100) * Z
    F     =   fun(  Z, LT)
    u_F   = u_fun(u_Z, LT)
    return Z, u_Z, F, u_F, fun, u_fun, LT


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


def test_correction_attributes_1d(toy_data_1d):
    X, E, Eu, F, Fu, correct = toy_data_1d
    assert_allclose(correct.xs[0], X ) # correct.xs is a list of axis
    assert_allclose(correct.fs   , F )
    assert_allclose(correct.us   , Fu)


def test_correction_attributes_1d_unnormalized(toy_data_1d):
    X, E, Eu, F, Fu, correct = toy_data_1d
    c = Correction((X,), F, Fu, False)
    assert_equal(c.fs, F )
    assert_equal(c.us, Fu)


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


def test_fcorrection(toy_f_data):
    Z, u_Z, F, u_F, fun, u_fun, LT = toy_f_data

    Z_test =   Z
    U_test = u_Z

    fcorr          = Fcorrection(fun, u_fun, (LT,), (LT,))
    f_test, u_test = fcorr(Z_test, U_test)

    assert_allclose(  F, f_test)
    assert_allclose(u_F, u_test)
    

"""
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