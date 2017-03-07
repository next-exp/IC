from __future__ import absolute_import

from   pytest                           import mark, fixture

import os
import numpy as np
import numpy.testing as npt
import tables as tb

import invisible_cities.core.dnn_functions as dnnf
from   invisible_cities.database        import load_db

def test_read_pmaps():

    # load the SiPM (x,y) values
    DataSensor = load_db.DataSiPM(0)
    xs = DataSensor.X.values
    ys = DataSensor.Y.values

    id_to_coords = {}
    for ID, x, y in zip(range(1792), xs, ys):
        id_to_coords[np.int32(ID)] = np.array([x, y])

    # read the test file
    PATH_IN = os.path.join(os.environ['ICDIR'],
           'database/test_data/',
           'kr_ACTIVE_7bar_MCRD.h5')
    print(PATH_IN)
    maps, energies, evt_numbers = dnnf.read_pmaps([PATH_IN], 10, id_to_coords, 3, 10000)

    # make several checks
    assert maps.shape == (5, 48, 48, 3)
    assert energies.shape == (5, 3)
    assert (evt_numbers == np.array([[0],[1],[2],[3],[4]])).all()

def test_read_xyz_labels():

    # read the test file
    PATH_IN = os.path.join(os.environ['ICDIR'],
           'database/test_data/',
           'kr_ACTIVE_7bar_MCRD.h5')
    evt_numbers = np.array([[0],[1],[2],[3],[4]])
    labels, levt_numbers = dnnf.read_xyz_labels([PATH_IN], 10, evt_numbers)

    # make several checks
    assert labels.shape == (5, 3)
    assert (levt_numbers == np.array([0, 1, 2, 3, 4])).all()
