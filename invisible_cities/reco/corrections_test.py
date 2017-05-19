from collections import namedtuple

import numpy as np

from numpy.testing import assert_allclose

from pytest import fixture

from . corrections import Correction

EnergyMap = namedtuple('EnergyMap',
                       'E_max, E_00, x_0, y_0, x, y, E')

@fixture(scope='session')
def toy_energy_map():
    E_max = 120
    E_00  =  30
    x_0   =  10.1
    y_0   =  20.3
    x = np.array([x_0, 20.7])
    y = np.array([y_0, 10.2, 10.3])
    E = np.array([[E_00,  40,  60],
                  [  80,  90, E_max]])
    return EnergyMap(E_max, E_00, x_0, y_0, x, y, E)



def test_energy_correction_max(toy_energy_map):
    m = toy_energy_map
    U = np.ones_like(m.E)
    correct = Correction(m.x, m.y, m.E, U, 'max')
    correction = m.E_00 / m.E_max
    assert_allclose(correct(m.x_0, m.y_0), correction)


def test_energy_correction_None(toy_energy_map):
    m = toy_energy_map
    U = np.ones_like(m.E)
    correct = Correction(m.x, m.y, m.E, U, None)
    correction = m.E_00 / m.E_max
    NO_CORRECTION = 1
    assert_allclose(correct(m.x_0, m.y_0), NO_CORRECTION)
