from operator    import mul
from operator    import truediv
from collections import namedtuple

import numpy as np

from pytest import fixture
from pytest import mark

from .  corrections        import Correction
from .  dst_functions      import load_xy_corrections
from .  dst_functions      import load_lifetime_xy_corrections

normalization_data = namedtuple("normalization_data", "node kwargs op")

@fixture(scope  = "session",
         params = [False, True])
def normalization(request):
    if request.param:
        node   = "LifetimeXY_inverse"
        kwargs = {"norm_strategy": "const",
                  "norm_opts"    : {"value": 1}}
        op     = truediv
    else:
        node   = "LifetimeXY"
        kwargs = {}
        op     = mul
    return normalization_data(node, kwargs, op)


def test_load_xy_corrections(corr_toy_data, normalization):
    filename, true_data = corr_toy_data
    x, y, E, U, _ = true_data
    corr          = load_xy_corrections(filename,
                                        node = normalization.node,
                                        **normalization.kwargs)
    assert corr == Correction((x,y), E, U)


@mark.parametrize("scale",
                  (0.5, 1, 2.0))
def test_load_lifetime_xy_corrections(corr_toy_data, normalization, scale):
    filename, true_data = corr_toy_data
    x, y, LT, U, _ = true_data
    corr           = load_lifetime_xy_corrections(filename,
                                                  node  = normalization.node,
                                                  scale = scale,
                                                  **normalization.kwargs)

    LT = normalization.op(LT, scale)
    U  = normalization.op(U , scale)
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
