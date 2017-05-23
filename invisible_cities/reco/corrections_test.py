import numpy as np

from ..core             import fit_functions as fitf
from ..reco.corrections import Correction, Fcorrection

from numpy.testing import assert_equal, assert_allclose
from pytest        import fixture, mark
from collections   import namedtuple

from hypothesis             import given
from hypothesis.strategies  import floats
from hypothesis.strategies  import integers
from hypothesis.strategies  import composite
from hypothesis.extra.numpy import arrays


data_1d = namedtuple("data_1d",   "z E prof")
data_2d = namedtuple("data_2d", "x y E prof")

FField_1d = namedtuple("Ffield"  , "X   P Pu F Fu fun u_fun correct")
EField_1d = namedtuple("Efield1d", "X   E Eu F Fu imax correct")
EField_2d = namedtuple("Efield2d", "X Y E Eu F Fu imax jmax correct")


@composite
def uniform_energy_1d(draw):
    size  = draw(integers(min_value= 2  , max_value=10 ))
    X0    = draw(floats  (min_value=-100, max_value=100))
    dX    = draw(floats  (min_value= 0.1, max_value=100))
    X     = np.arange(size) * dX + X0
    E     = draw(arrays(float, size, floats(min_value=1e+3, max_value=2e+3)))
    u_rel = draw(arrays(float, size, floats(min_value=1e-2, max_value=2e-1)))
    Eu    = E * u_rel

    i_max = np.argmax(E)
    e_max = E [i_max]
    u_max = Eu[i_max]

    F     = E.max()/E
    Fu    = F * (Eu**2/E**2 + u_max**2/e_max**2)**0.5

    corr  = Correction((X,), E, Eu, "max")
    return EField_1d(X, E, Eu, F, Fu, i_max, corr)


@composite
def uniform_energy_2d(draw, interp_strategy="nearest"):
    x_size  = draw(integers(min_value=2   , max_value=10 ))
    y_size  = draw(integers(min_value=2   , max_value=10 ))
    X0      = draw(floats  (min_value=-100, max_value=100))
    Y0      = draw(floats  (min_value=-100, max_value=100))
    dX      = draw(floats  (min_value= 0.1, max_value=100))
    dY      = draw(floats  (min_value= 0.1, max_value=100))
    X       = np.arange(x_size) * dX + X0
    Y       = np.arange(y_size) * dY + Y0
    E       = draw(arrays(float, (x_size, y_size), floats(min_value = 1e3 , max_value = 2e3 )))
    u_rel   = draw(arrays(float, (x_size, y_size), floats(min_value = 1e-2, max_value = 2e-1)))
    Eu      = E * u_rel

    i_max = draw(integers(min_value=0, max_value=x_size-1))
    j_max = draw(integers(min_value=0, max_value=y_size-1))
    e_max = E [i_max, j_max]
    u_max = Eu[i_max, j_max]

    F     = e_max/E
    Fu    = F * (Eu**2/E**2 + u_max**2/e_max**2)**0.5

    corr  = Correction((X,Y), E, Eu,
                         norm_strategy = "index", index = (i_max, j_max),
                       interp_strategy = interp_strategy)
    return EField_2d(X, Y, E, Eu, F.flatten(), Fu.flatten(), i_max, j_max, corr)


@composite
def uniform_energy_fun_data_1d(draw):
    fun   = lambda x, LT: fitf.expo(x, 1, -LT)
    u_fun = lambda x, LT: x/LT**2 * fun(x, LT)
    LT    = draw(floats(min_value=1e2, max_value=1e3))
    u_LT  = draw(floats(min_value=1e-2, max_value=1e-1)) * LT
    X     = np.linspace(0, 600, 100)
    F     =   fun(X, LT)
    u_F   = u_fun(X, LT)
    corr  = Fcorrection(fun, u_fun, (LT,), (LT,))
    return FField_1d(X, LT, u_LT, F, u_F, fun, u_fun, corr)


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


#--------------------------------------------------------
#--------------------------------------------------------

@given(uniform_energy_1d())
def test_correction_attributes_1d(toy_data_1d):
    X, _, _, F, Fu, _, correct = toy_data_1d
    assert_allclose(correct.xs[0], X ) # correct.xs is a list of axis
    assert_allclose(correct.fs   , F )
    assert_allclose(correct.us   , Fu)


@given(uniform_energy_1d())
def test_correction_attributes_1d_unnormalized(toy_data_1d):
    X, _, _, F, Fu, _, correct = toy_data_1d
    c = Correction((X,), F, Fu, False)
    assert_allclose(c.fs, F )
    assert_allclose(c.us, Fu)


@given(uniform_energy_1d())
def test_correction_call_1d(toy_data_1d):
    X, E, Eu, F, Fu, _, correct = toy_data_1d

    X_test = X
    F_test, U_test = correct(X_test)
    assert_allclose(F_test, F )
    assert_allclose(U_test, Fu)


@given(uniform_energy_1d())
def test_correction_normalization(toy_data_1d):
    X, *_, i_max, correct = toy_data_1d
    X_test  = X[i_max]
    assert_allclose(correct(X_test).value, 1) # correct.xs is a list of axis


#--------------------------------------------------------

@given(uniform_energy_2d())
def test_correction_attributes_2d(toy_data_2d):
    *_, F, Fu, _, _, correct = toy_data_2d
    # attributes of the Correction class are 2d arrays,
    # so they must be flatten for comparison
    assert_allclose(correct.fs.flatten(), F )
    assert_allclose(correct.us.flatten(), Fu)


@given(uniform_energy_2d())
def test_correction_attributes_2d_unnormalized(toy_data_2d):
    X, Y, _, _, F, Fu, _, _, correct = toy_data_2d
    c = Correction((X,Y), F, Fu, False)
    assert_allclose(c.fs, F )
    assert_allclose(c.us, Fu)


@given(uniform_energy_2d())
def test_correction_call_2d(toy_data_2d):
    X, Y, _, _, F, Fu, _, _, correct = toy_data_2d

    # Combine x and y so they are paired
    X_test = np.repeat(X, Y.size)
    Y_test = np.tile  (Y, X.size)
    F_test, U_test = correct(X_test, Y_test)
    assert_allclose(F_test, F )
    assert_allclose(U_test, Fu)


#--------------------------------------------------------

@given(uniform_energy_fun_data_1d())
def test_fcorrection(toy_f_data):
    Z, LT, u_LT, F, u_F, fun, u_fun, correct = toy_f_data

    Z_test         = Z
    f_test, u_test = correct(Z_test)

    assert_allclose(  F, f_test)
    assert_allclose(u_F, u_test)
    

"""
@mark.slow
def test_corrections_1d(gauss_data_1d):
    zevt, Eevt, (z, E, Ee) = gauss_data_1d
    zcorr = Correction((z,), E, Ee, "max")
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
    xycorr = Correction((x, y), E, Ee, "index", index=(25,25))
    Eevt  *= xycorr(xevt, yevt)[0]

    mean = np.std(Eevt)
    std  = np.std(Eevt)
    y, x = np.histogram(Eevt, np.linspace(mean - 3 * std, mean + 3 * std, 100))
    x = x[:-1] + np.diff(x) * 0.5
    f = fitf.fit(fitf.gauss, x, y, (1e5, mean, std))
    assert f.chi2 < 3
"""