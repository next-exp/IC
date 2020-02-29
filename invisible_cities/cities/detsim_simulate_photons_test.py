import os
import numpy  as np
import tables as tb
import pandas as pd
from pytest import fixture

import invisible_cities.database.load_db as db

from invisible_cities.io.mcinfo_io import read_mchits_df

from invisible_cities.cities.detsim_simulate_photons import *

## same as in detsim_simulate_electrons_test
@fixture(params = ["Kr83_nexus_v5_03_00_ACTIVE_7bar_3evts.MCRD.h5"])
def nexusMC_filename(request, ICDATADIR):
    return os.path.join(ICDATADIR, request.param)

## same as in detsim_simulate_electrons_test
@fixture
def random_event_hits(nexusMC_filename):
    filename = nexusMC_filename

    with tb.open_file(filename) as h5in:
        extents = pd.read_hdf(filename, 'MC/extents')
        event_ids  = extents.evt_number
        hits_df    = read_mchits_df(h5in, extents)

        evt = np.random.choice(event_ids)
        hits = hits_df.loc[evt, :, :]

        xs = hits["x"].values
        ys = hits["y"].values
        zs = hits["z"].values
        energies = hits["energy"].values

    return xs, ys, zs, energies


@fixture
def sensor_data():
    pmtdata = db.DataPMT("new", -1)
    return pmtdata


def test_pes_at_sensors_shape(random_event_hits, sensor_data):

    #### this could be set to other psf with detsim_get_psf functions
    psf = lambda x, y, z=0: x+y

    x_sensors, y_sensors = sensor_data["X"].values, sensor_data["Y"].values
    z_sensors = 100

    xs, ys, zs, energies = random_event_hits
    photons = generate_s1_photons(energies, 10)

    pes = pes_at_sensors(xs, ys, zs, photons,
                         x_sensors, y_sensors, z_sensors,
                         psf)

    nsensors = len(sensor_data)
    nhits    = len(xs)

    assert pes.shape == (nsensors, nhits)
