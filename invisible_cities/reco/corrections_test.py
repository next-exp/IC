from collections import namedtuple

import numpy as np

from numpy.testing import assert_allclose

from pytest import fixture

from . corrections import Correction

EnergyMap = namedtuple('EnergyMap',
                       'E, U, E_max, U_max, E_00, U_01, x_0, y_0, y_1, x, y')

@fixture(scope='session')
def toy_energy_map():
    E_max = 120
    U_max =   6
    E_00  =  30
    U_01  =   5
    x_0   =  10.1
    y_0   =  20.3
    y_1   =  10.2
    x = np.array([x_0, 20.7])
    y = np.array([y_0, y_1, 10.3])
    E = np.array([[E_00,  40,  60],
                  [  80,  90, E_max]])
    U = np.array([[   3 ,U_01,  12],
                  [   4,   9, U_max]])

    return EnergyMap(E      = E,
                     U      = U,
                     E_max  = E_max,
                     U_max  = U_max,
                     E_00   = E_00,
                     U_01   = U_01,
                     x_0    = x_0,
                     y_0    = y_0,
                     y_1    = y_1,
                     x      = x,
                     y      = y)


def test_energy_correction_max(toy_energy_map):
    m = toy_energy_map
    correct = Correction(m.x, m.y, m.E, m.U, 'max')
    CORRECTION = m.E_00 / m.E_max
    assert_allclose(correct.E(m.x_0, m.y_0), CORRECTION)


def test_energy_correction_None(toy_energy_map):
    m = toy_energy_map
    correct = Correction(m.x, m.y, m.E, m.U, None)
    NO_CORRECTION = 1
    assert_allclose(correct.E(m.x_0, m.y_0), NO_CORRECTION)


def test_energy_uncertainty_correction_None(toy_energy_map):
    m = toy_energy_map
    correct = Correction(m.x, m.y, m.E, m.U, None)
    NO_CORRECTION = 1
    assert_allclose(correct.U(m.x_0, m.y_1), NO_CORRECTION)
