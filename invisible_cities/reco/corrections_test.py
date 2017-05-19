import numpy as np

from numpy.testing import assert_allclose

from . corrections import Correction

def test_correction():
    E_max = 120
    E_00  =  30
    x_0   =  10.1
    y_0   =  20.3
    x = np.array([x_0, 20.7])
    y = np.array([y_0, 10.2, 10.3])
    E = np.array([[E_00,  40,  60],
                  [  80,  90, E_max]])

    U = np.ones_like(E)

    correct = Correction(x, y, E, U, 'max')

    correction = E_00 / E_max
    assert_allclose(correct(x_0, y_0), correction)
