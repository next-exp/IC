import numpy as np

from .  corrections        import Correction
from .  dst_functions      import load_xy_corrections
from .  dst_functions      import load_lifetime_xy_corrections


def test_load_xy_corrections(corr_toy_data):
    filename, true_data = corr_toy_data
    x, y, E, U, _ = true_data
    corr          = load_xy_corrections(filename)
    assert corr == Correction((x,y), E, U)


def test_load_lifetime_xy_corrections(corr_toy_data):
    filename, true_data = corr_toy_data
    x, y, LT, U, _ = true_data
    corr           = load_lifetime_xy_corrections(filename)


    for i in np.linspace(0, 2, 5):
        # This should yield exp(i * x/x) = exp(i)
        z_test   = LT.flatten() * i
        x_test   = np.repeat(x, y.size)
        y_test   = np.tile  (y, x.size)
        (f_test,
         u_test) = corr(z_test, x_test, y_test)

        f_true = np.exp(i)
        u_true = z_test * U.flatten()/LT.flatten()**2 * f_test
        assert np.allclose(f_test, f_true)
        assert np.allclose(u_test, u_true)
