from collections import namedtuple

import numpy as np

from .. core                 import fit_functions as fitf
from .. core .stat_functions import poisson_sigma
from .. core .exceptions     import ParameterNotSet
from .. icaro.hst_functions  import shift_to_bin_centers
from .. reco .corrections    import Correction
from .. reco .corrections    import Fcorrection
from .. reco .corrections    import LifetimeCorrection
from .. reco .corrections    import LifetimeRCorrection
from .. reco .corrections    import LifetimeXYCorrection
from .. reco .corrections    import opt_nearest
from .. reco .corrections    import opt_linear
from .. reco .corrections    import opt_cubic

from numpy.testing import assert_allclose
from pytest        import fixture
from pytest        import mark
from pytest        import raises
from flaky         import flaky

from hypothesis             import given
from hypothesis             import settings
from hypothesis.strategies  import floats
from hypothesis.strategies  import integers
from hypothesis.strategies  import composite
from hypothesis.extra.numpy import arrays


data_1d = namedtuple("data_1d",   "X   E Eu Xdata       Edata")
data_2d = namedtuple("data_2d",   "X Y E Eu Xdata Ydata Edata")

FField_1d = namedtuple("Ffield1d", "X   P Pu F Fu fun u_fun")
FField_2d = namedtuple("Ffield2d", "X Y P Pu F Fu fun u_fun")
EField_1d = namedtuple("Efield1d", "X   E Eu F Fu imax"     )
EField_2d = namedtuple("Efield2d", "X Y E Eu F Fu imax jmax")


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
    Fu    = F * (Eu**2 / E**2 + u_max**2 / e_max**2)**0.5

    return EField_1d(X, E, Eu, F, Fu, i_max)


@composite
def uniform_energy_2d(draw):
    x_size  = draw(integers(min_value=2   , max_value=10 ))
    y_size  = draw(integers(min_value=2   , max_value=10 ))
    X0      = draw(floats  (min_value=-100, max_value=100))
    Y0      = draw(floats  (min_value=-100, max_value=100))
    dX      = draw(floats  (min_value= 0.1, max_value=100))
    dY      = draw(floats  (min_value= 0.1, max_value=100))
    X       = np.arange(x_size) * dX + X0
    Y       = np.arange(y_size) * dY + Y0
    E       = draw(arrays(float, (x_size, y_size), floats(min_value = 1e+3,
                                                          max_value = 2e+3)))
    u_rel   = draw(arrays(float, (x_size, y_size), floats(min_value = 1e-2,
                                                          max_value = 2e-1)))
    Eu      = E * u_rel

    i_max = draw(integers(min_value=0, max_value=x_size - 1))
    j_max = draw(integers(min_value=0, max_value=y_size - 1))
    e_max = E [i_max, j_max]
    u_max = Eu[i_max, j_max]

    F     = e_max / E
    Fu    = F * (Eu**2 / E**2 + u_max**2 / e_max**2)**0.5

    return EField_2d(X, Y, E, Eu, F.flatten(), Fu.flatten(), i_max, j_max)


@composite
def uniform_energy_fun_data_1d(draw):
    fun   = lambda z, LT, u_LT: fitf.expo(z, 1, LT)
    u_fun = lambda z, LT, u_LT: z * u_LT / LT**2 * fun(z, LT, u_LT)
    LT    = draw(floats(min_value=1e+2, max_value=1e+3))
    u_LT  = draw(floats(min_value=1e+1, max_value=1e+1))
    Z     = np.linspace(0, 600, 100)
    F     =   fun(Z, LT, u_LT)
    u_F   = u_fun(Z, LT, u_LT)
    return FField_1d(Z, LT, u_LT, F, u_F, fun, u_fun)


@composite
def uniform_energy_fun_data_2d(draw):
    def fun(z, r, a, b, c, u_a, u_b, u_c):
        LT = a - b * r * np.exp(r / c)
        return fitf.expo(z, 1, LT)

    def u_fun(z, r, a, b, c, u_a, u_b, u_c):
        LT   = a - b * r * np.exp(r / c)
        u_LT = (u_a**2 + u_b**2 * np.exp(2 * r / c) +
                u_c**2 * b**2 * r**2 * np.exp(2 * r / c) / c**4)**0.5
        return z * u_LT / LT**2 * fun(z, r, a, b, c, u_a, u_b, u_c)

    a     = draw(floats(min_value=1e+2, max_value=1e+3));u_a = 0.1 * a
    b     = draw(floats(min_value=1e-2, max_value=1e-1));u_b = 0.1 * b
    c     = draw(floats(min_value=1e+3, max_value=5e+3));u_c = 0.1 * c
    Z     = np.linspace(0, 600, 100)
    R     = np.linspace(0, 200, 100)
    F     =   fun(Z, R, a, b, c, u_a, u_b, u_c)
    u_F   = u_fun(Z, R, a, b, c, u_a, u_b, u_c)
    return FField_2d(Z, R, (a, b, c), (u_a, u_b, u_c), F, u_F, fun, u_fun)


@composite
def uniform_energy_fun_data_3d(draw):
    x_size  = draw(integers(min_value= 2  , max_value=10 ))
    y_size  = draw(integers(min_value= 2  , max_value=10 ))
    X0      = draw(floats  (min_value=-100, max_value=100))
    Y0      = draw(floats  (min_value=-100, max_value=100))
    dX      = draw(floats  (min_value= 0.1, max_value=100))
    dY      = draw(floats  (min_value= 0.1, max_value=100))
    X       = np.arange(x_size) * dX + X0
    Y       = np.arange(y_size) * dY + Y0

    LTs     = draw(arrays(float, (x_size, y_size), floats(min_value = 1e+2,
                                                          max_value = 1e+3)))
    u_LTs   = LTs * 0.1

    LTc     = Correction((X, Y), LTs, u_LTs,
                         **opt_nearest)

    def LT_corr(z, x, y):
        return np.exp(z / LTc(x, y).value)

    def u_LT_corr(z, x, y):
        ltc = LTc(x, y)
        return z * ltc.uncertainty / ltc.value**2 * np.exp(z / ltc.value)

    return FField_2d(X, Y, LTs, u_LTs, LTs, u_LTs, LT_corr, u_LT_corr)


@fixture
def gauss_data_1d():
    mean = lambda z: 1e4 * np.exp(-z / 1000)
    Nevt = 100000
    Zevt = np.random.uniform(0, 500, size=Nevt)
    Eevt = np.random.normal(mean(Zevt), mean(Zevt)**0.5)
    prof = fitf.profileX(Zevt, Eevt, 50, (0, 500))
    return data_1d(*prof, Zevt, Eevt)


@fixture
def gauss_data_2d():
    mean = lambda x, y: 1e4 * np.exp(-(x**2 + y**2) / 400**2)
    Nevt = 100000
    Xevt = np.random.uniform(-200, 200, size=Nevt)
    Yevt = np.random.uniform(-200, 200, size=Nevt)
    Eevt = np.random.normal(mean(Xevt, Yevt), mean(Xevt, Yevt)**0.5)
    prof = fitf.profileXY(Xevt, Yevt, Eevt, 50, 50, (-200, 200), (-200, 200))
    return data_2d(*prof, Xevt, Yevt, Eevt)


#--------------------------------------------------------
#--------------------------------------------------------
@mark.parametrize("strategy options".split(),
                  (("const", {}),
                   ("index", {}),
                   ("const", {"wrong_option": None}),
                   ("index", {"wrong_option": None})))
def test_correction_raises_exception_when_input_is_incomplete(strategy, options):
    data = np.arange(5)
    with raises(ParameterNotSet):
        Correction((data,), data, data,
                   norm_strategy = strategy,
                   norm_opts     = options,
                   **opt_nearest)


def test_correction_raises_exception_when_data_is_invalid():
    x   = np.arange(  0, 10)
    y   = np.arange(-10,  0)
    z   = np.zeros ((x.size, y.size))
    u_z = np.ones  ((x.size, y.size))
    with raises(AssertionError):
        Correction((x, y), z, u_z,
                   norm_strategy =  "index",
                   norm_opts     = {"index": (0, 0)},
                   **opt_nearest)


@given(uniform_energy_1d())
def test_correction_attributes_1d(toy_data_1d):
    X, E, Eu, F, Fu, _ = toy_data_1d
    correct  = Correction((X,), E, Eu,
                          norm_strategy = "max",
                          **opt_nearest)
    assert_allclose(correct._xs[0], X ) # correct.xs is a list of axis
    assert_allclose(correct._fs   , F )
    assert_allclose(correct._us   , Fu)


@given(uniform_energy_1d())
def test_correction_attributes_1d_unnormalized(toy_data_1d):
    X, _, _, F, Fu, _ = toy_data_1d
    c = Correction((X,), F, Fu,
                   norm_strategy = None,
                   **opt_nearest)
    assert_allclose(c._fs, F )
    assert_allclose(c._us, Fu)


@settings(max_examples=1)
@given(uniform_energy_1d())
def test_correction_call_scalar_values_1d(toy_data_1d):
    X, E, Eu, F, Fu, _ = toy_data_1d
    correct  = Correction((X,), E, Eu,
                          norm_strategy = "max",
                          **opt_nearest)
    F_corrected, U_corrected = correct(X[0])
    assert F_corrected == F [0]
    assert U_corrected == Fu[0]


@given(uniform_energy_1d())
def test_correction_call_1d(toy_data_1d):
    X, E, Eu, F, Fu, _ = toy_data_1d
    correct  = Correction((X,), E, Eu,
                          norm_strategy = "max",
                          **opt_nearest)
    F_corrected, U_corrected = correct(X)
    assert_allclose(F_corrected, F )
    assert_allclose(U_corrected, Fu)


@given(uniform_energy_1d())
def test_correction_normalization_1d_to_max(toy_data_1d):
    X, E, Eu, *_, i_max = toy_data_1d
    correct  = Correction((X,), E, Eu,
                          norm_strategy = "max",
                          **opt_nearest)

    x_test = X
    corrected_E = E * correct(x_test).value
    assert_allclose(corrected_E, np.max(E))


@given(uniform_energy_1d(),
       floats  (min_value=1e-8, max_value=1e8))
def test_correction_normalization_1d_to_const(toy_data_1d, norm_value):
    X, E, Eu, _, _, _ = toy_data_1d
    c = Correction((X,), E, Eu,
                   norm_strategy = "const",
                   norm_opts     = {"value": norm_value},
                   **opt_nearest)

    assert_allclose(c._fs, norm_value / E)
    assert_allclose(c._us, norm_value / E**2 * Eu)


@given(uniform_energy_1d())
def test_correction_normalization_to_center_1d(toy_data_1d):
    X, E, Eu, *_ = toy_data_1d
    c = Correction((X,), E, Eu,
                   norm_strategy = "center")

    norm_index = X.size // 2
    norm_value = E [norm_index]
    norm_uncer = Eu[norm_index]
    prop_uncer = (Eu / E)**2 + (norm_uncer / norm_value)**2
    prop_uncer = prop_uncer**0.5 * norm_value / E

    assert_allclose(c._fs, norm_value / E)
    assert_allclose(c._us, prop_uncer    )


@given(uniform_energy_2d())
def test_correction_normalization_to_center_2d(toy_data_2d):
    X, Y, E, Eu, *_ = toy_data_2d
    c = Correction((X, Y), E, Eu,
                   norm_strategy = "center")

    norm_index = X.size // 2, Y.size // 2
    norm_value = E [norm_index]
    norm_uncer = Eu[norm_index]
    prop_uncer = (Eu / E)**2 + (norm_uncer / norm_value)**2
    prop_uncer = prop_uncer**0.5 * norm_value / E

    assert_allclose(c._fs, norm_value / E)
    assert_allclose(c._us, prop_uncer    )


#--------------------------------------------------------

@given(uniform_energy_2d())
def test_correction_attributes_2d(toy_data_2d):
    X, Y, E, Eu, F, Fu, i_max, j_max = toy_data_2d
    correct = Correction((X, Y), E, Eu,
                         norm_strategy =  "index",
                         norm_opts     = {"index": (i_max, j_max)},
                         **opt_nearest)

    # attributes of the Correction class are 2d arrays,
    # so they must be flatten for comparison
    assert_allclose(correct._fs.flatten(), F )
    assert_allclose(correct._us.flatten(), Fu)


@given(uniform_energy_2d())
def test_correction_attributes_2d_unnormalized(toy_data_2d):
    X, Y, _, _, F, Fu, _, _ = toy_data_2d
    c = Correction((X, Y), F, Fu,
                   norm_strategy = None,
                   **opt_nearest)

    assert_allclose(c._fs, F )
    assert_allclose(c._us, Fu)


@settings(max_examples=1)
@given(uniform_energy_2d())
def test_correction_call_scalar_values_2d(toy_data_2d):
    X, Y, E, Eu, F, Fu, i_max, j_max = toy_data_2d
    correct = Correction((X,Y), E, Eu,
                         norm_strategy =  "index",
                         norm_opts     = {"index": (i_max, j_max)},
                         **opt_nearest)

    F_corrected, U_corrected = correct(X[0], Y[0])
    assert F_corrected == F [0]
    assert U_corrected == Fu[0]


@given(uniform_energy_2d())
def test_correction_normalization_2d_to_max(toy_data_2d):
    X, Y, E, Eu, *_, i_max = toy_data_2d
    correct  = Correction((X, Y), E, Eu,
                          norm_strategy = "max")

    x_test      = np.repeat(X, Y.size)
    y_test      = np.tile  (Y, X.size)
    corrected_E = E.flatten() * correct(x_test, y_test).value
    assert_allclose(corrected_E, np.max(E))


@given(uniform_energy_2d())
def test_correction_call_2d(toy_data_2d):
    X, Y, E, Eu, F, Fu, i_max, j_max = toy_data_2d
    correct = Correction((X, Y), E, Eu,
                         norm_strategy =  "index",
                         norm_opts     = {"index": (i_max, j_max)},
                         **opt_nearest)

    # create a collection of (x,y) point such that the
    # x coordinates are stored in X_sample and the y coordinates in Y_sample
    X_sample = np.array([x for x in X for _ in Y])
    Y_sample = np.array([y for _ in X for y in Y])

    F_corrected, U_corrected = correct(X_sample, Y_sample)
    assert_allclose(F_corrected, F )
    assert_allclose(U_corrected, Fu)


#--------------------------------------------------------

@given(uniform_energy_fun_data_1d())
def test_fcorrection(toy_f_data):
    Z, LT, u_LT, F, u_F, fun, u_fun = toy_f_data
    correct = Fcorrection(fun, u_fun, (LT, u_LT))
    f_corrected, u_corrected = correct(Z)

    assert_allclose(  F, f_corrected)
    assert_allclose(u_F, u_corrected)


@given(uniform_energy_fun_data_1d())
def test_lifetimecorrection(toy_f_data):
    Z, LT, u_LT, F, u_F, fun, u_fun = toy_f_data
    correct = LifetimeCorrection(LT, u_LT)
    f_corrected, u_corrected = correct(Z)

    assert_allclose(  F, f_corrected)
    assert_allclose(u_F, u_corrected)


@given(uniform_energy_fun_data_2d())
def test_lifetimeRcorrection(toy_f_data):
    Z, R, pars, u_pars, F, u_F, fun, u_fun = toy_f_data
    correct = LifetimeRCorrection(pars, u_pars)
    f_corrected, u_corrected = correct(Z, R)

    assert_allclose(  F, f_corrected)
    assert_allclose(u_F, u_corrected)


@given(uniform_energy_fun_data_3d())
def test_lifetimeXYcorrection(toy_f_data):
    Xgrid, Ygrid, LTs, u_LTs, LTs, u_LTs, LT_corr, u_LT_corr = toy_f_data

    X       = np.repeat  (Xgrid, Ygrid.size)
    Y       = np.tile    (Ygrid, Xgrid.size)
    Z       = np.linspace(0, 50, X    .size)
    F, u_F  = LT_corr(Z, X, Y), u_LT_corr(Z, X, Y)
    correct = LifetimeXYCorrection(LTs, u_LTs, Xgrid, Ygrid, **opt_nearest)
    f_corrected, u_corrected = correct(Z, X, Y)

    assert_allclose(  F, f_corrected)
    assert_allclose(u_F, u_corrected)


@given(uniform_energy_fun_data_3d())
def test_lifetimeXYcorrection_kwargs(toy_f_data):
    Xgrid, Ygrid, LTs, u_LTs, LTs, u_LTs, LT_corr, u_LT_corr = toy_f_data
    kwargs = {"norm_strategy" :  "const",
              "norm_opts"     : {"value": 1},
              **opt_nearest}

    X       = np.repeat  (Xgrid, Ygrid.size)
    Y       = np.tile    (Ygrid, Xgrid.size)
    Z       = np.linspace(0, 50, X    .size)
    F, u_F  = LT_corr(Z, X, Y), u_LT_corr(Z, X, Y)

    # These input values are chosen because they
    # effectively cancel the normalization.
    correct = LifetimeXYCorrection(1 / LTs, u_LTs / LTs**2,
                                   Xgrid, Ygrid, **kwargs)
    f_corrected, u_corrected = correct(Z, X, Y)

    assert_allclose(  F, f_corrected)
    assert_allclose(u_F, u_corrected)


#--------------------------------------------------------


@mark.slow
@flaky(max_runs=5, min_passes=4)
def test_corrections_1d(gauss_data_1d):
    Z, E, Eu, Zevt, Eevt = gauss_data_1d

    correct = Correction((Z,), E, Eu,
                         norm_strategy = "max",
                         **opt_nearest)
    Eevt   *= correct(Zevt).value

    mean = np.mean(Eevt)
    std  = np.std (Eevt)

    y, x = np.histogram(Eevt, np.linspace(mean - 3 * std,
                                          mean + 3 * std,
                                          100))
    x     = shift_to_bin_centers(x)
    sigma = poisson_sigma(y)
    f     = fitf.fit(fitf.gauss, x, y, (1e5, mean, std), sigma=sigma)

    assert 0.75 < f.chi2 < 1.5


@mark.slow
@flaky(max_runs=5, min_passes=4)
def test_corrections_2d(gauss_data_2d):
    X, Y, E, Eu, Xevt, Yevt, Eevt = gauss_data_2d
    correct = Correction((X, Y), E, Eu,
                         norm_strategy =  "index",
                         norm_opts     = {"index": (25, 25)},
                         **opt_nearest)
    Eevt   *= correct(Xevt, Yevt)[0]

    mean = np.mean(Eevt)
    std  = np.std (Eevt)

    y, x = np.histogram(Eevt, np.linspace(mean - 3 * std,
                                          mean + 3 * std,
                                          100))
    x     = shift_to_bin_centers(x)
    sigma = poisson_sigma(y)
    f     = fitf.fit(fitf.gauss, x, y, (1e5, mean, std), sigma=sigma)

    assert 0.75 < f.chi2 < 1.5


def test_corrections_linear_interpolation():
    # This is the function f(x,y) = x + y on a square grid. Because the
    # interpolation is linear, any point with coordinates (x, y) should
    # yield exactly f(x, y).
    xmin, xmax  = 10, 20
    ymin, ymax  = 20, 30
    grid_x      = np.arange(xmin, xmax+1)
    grid_y      = np.arange(ymin, ymax+1)
    grid_points = np.array([(i, j) for i in grid_x\
                                   for j in grid_y])

    grid_fun    = np.sum
    grid_values = np.apply_along_axis(grid_fun, 1, grid_points)
    grid_uncert = np.apply_along_axis(grid_fun, 1, grid_points)/10

    correct = Correction((grid_x, grid_y),
                         grid_values,
                         grid_uncert,
                         **opt_linear)

    x_test  = np.random.uniform(xmin, xmax, size=100)
    y_test  = np.random.uniform(ymin, ymax, size=100)
    xy_test = np.stack([x_test, y_test], axis=1)

    correction      = correct(x_test, y_test)
    expected_values = np.apply_along_axis(grid_fun, 1, xy_test)
    expected_uncert = expected_values/10

    assert np.allclose(correction.value      , expected_values)
    assert np.allclose(correction.uncertainty, expected_uncert)


def test_corrections_cubic_interpolation():
    # This is the function f(x,y) = x + y on a square grid. Because the
    # interpolation is cubic, any point with coordinates (x, y) should
    # yield exactly f(x, y). This test should probably contain a more
    # complicated function.
    xmin, xmax  = 10, 20
    ymin, ymax  = 20, 30
    grid_x      = np.arange(xmin, xmax + 1)
    grid_y      = np.arange(ymin, ymax + 1)
    grid_points = np.array([(i, j) for i in grid_x\
                                   for j in grid_y])

    grid_fun    = np.sum
    grid_values = np.apply_along_axis(grid_fun, 1, grid_points)
    grid_uncert = np.apply_along_axis(grid_fun, 1, grid_points)/10

    correct = Correction((grid_x, grid_y),
                         grid_values,
                         grid_uncert,
                         **opt_cubic)

    x_test  = np.random.uniform(xmin, xmax, size=100)
    y_test  = np.random.uniform(ymin, ymax, size=100)
    xy_test = np.stack([x_test, y_test], axis=1)

    correction      = correct(x_test, y_test)
    expected_values = np.apply_along_axis(grid_fun, 1, xy_test)
    expected_uncert = expected_values/10

    assert np.allclose(correction.value      , expected_values)
    assert np.allclose(correction.uncertainty, expected_uncert)
