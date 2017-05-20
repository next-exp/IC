from collections import namedtuple

import numpy as np

from numpy.testing import assert_allclose

from pytest import fixture

from hypothesis             import given
from hypothesis.strategies  import floats
from hypothesis.strategies  import integers
from hypothesis.strategies  import composite
from hypothesis.extra.numpy import arrays

from . corrections import XYCorrection

EField = namedtuple('EField', 'i,j, x,y, E, U, ni,nj')

@composite
def energy_field(draw):
    x_size = draw(integers(min_value=2, max_value=10))
    y_size = draw(integers(min_value=2, max_value=20))
    dx     = draw(floats(min_value=0.1, max_value=0.7))
    dy     = draw(floats(min_value=0.1, max_value=0.7))
    x0     = draw(floats(min_value=-10, max_value=10))
    y0     = draw(floats(min_value=-10, max_value=10))
    E      = draw(arrays(float, (x_size, y_size), floats(min_value =   0.1,
                                                         max_value = 100)))
    U_rel  = draw(arrays(float, (x_size, y_size), floats(min_value =   0.01,
                                                         max_value =   0.1 )))
    U = E * U_rel
    x = np.arange(x_size) * dx + x0
    y = np.arange(y_size) * dy + y0
    i = draw(integers(min_value=0, max_value=x_size-1))
    j = draw(integers(min_value=0, max_value=y_size-1))

    ni = draw(integers(min_value=0, max_value=x_size-1))
    nj = draw(integers(min_value=0, max_value=y_size-1))

    return EField(i,j, x,y, E, U, ni,nj)


@given(energy_field())
def test_energy_correction_max(data):
    i,j, x,y, E, U, *_ = data
    correct = XYCorrection(x, y, E, U, 'max')
    E_CORRECTION = E[i,j] / E.max()
    assert_allclose(correct.E(x[i], y[j]), E_CORRECTION)


@given(energy_field())
def test_energy_uncertainty_correction_max(data):
    i,j, x,y, E, U, *_ = data
    correct = XYCorrection(x, y, E, U, 'max')
    E_max = E.max()
    U_CORRECTION = U[i,j] / E_max
    assert_allclose(correct.U(x[i], y[j]), U_CORRECTION)


@given(energy_field())
def test_energy_correction_index(data):
    i,j, x,y, E, U, ni, nj = data
    correct = XYCorrection(x, y, E, U, 'index', ni, nj)
    E_CORRECTION = E[i,j] / E[ni,nj]
    assert_allclose(correct.E(x[i], y[j]), E_CORRECTION)


@given(energy_field())
def test_energy_uncertainty_correction_index(data):
    i,j, x,y, E, U, ni, nj = data
    correct = XYCorrection(x, y, E, U, 'index', ni, nj)
    E_max = E.max()
    U_CORRECTION = U[i,j] / E[ni,nj]
    assert_allclose(correct.U(x[i], y[j]), U_CORRECTION)


@given(energy_field())
def test_energy_correction_None(data):
    i,j, x,y, E, U, *_ = data
    correct = XYCorrection(x, y, E, U, None)
    NO_CORRECTION = 1
    assert_allclose(correct.E(x[i], y[j]), NO_CORRECTION)


@given(energy_field())
def test_energy_uncertainty_correction_None(data):
    i,j, x,y, E, U, *_ = data
    correct = XYCorrection(x, y, E, U, None)
    NO_CORRECTION = 1
    assert_allclose(correct.U(x[i], y[j]), NO_CORRECTION)
