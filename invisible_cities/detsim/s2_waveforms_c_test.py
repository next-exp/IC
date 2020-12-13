import os
import pytest

import numpy as np

from pytest import warns

from .. database.load_db import DataSiPM

from .. core.system_of_units import *

from . light_tables_c  import LT_SiPM
from . light_tables_c  import LT_PMT
from . s2_waveforms_c  import create_wfs

from hypothesis                import given
from hypothesis                import settings
from hypothesis.strategies     import floats
from hypothesis.strategies     import integers
from hypothesis.extra.numpy    import arrays


@given(xs=arrays(np.float, 10, elements = floats  (min_value = -500*mm , max_value = 500*mm )),
       ys=arrays(np.float, 10, elements = floats  (min_value = -500*mm , max_value = 500*mm )),
       ts=arrays(np.float, 10, elements = floats  (min_value =    2*mus, max_value = 100*mus)),
       ps=arrays(np.int32, 10, elements = integers(min_value =    10   , max_value = 100    )))
def test_create_wfs_sipms_shape(get_dfs, xs, ys, ts, ps):
    datasipm = DataSiPM('new')
    fname, psf_df, psf_conf = get_dfs['psf']
    el_drift_velocity = 2.5 * mm/mus
    sensor_time_bin   = 1   * mus
    buffer_length     = 200 * mus
    lt = LT_SiPM(fname=fname, sipm_database=datasipm)
    n_sensors = lt.num_sensors
    waveform = create_wfs(xs, ys, ts, ps, lt, el_drift_velocity, sensor_time_bin, buffer_length)
    assert isinstance(waveform, np.ndarray)
    assert waveform.shape ==  (n_sensors, buffer_length//sensor_time_bin)


@given(xs=arrays(np.float, 10, elements = floats  (min_value = -500*mm , max_value = 500*mm )),
       ys=arrays(np.float, 10, elements = floats  (min_value = -500*mm , max_value = 500*mm )),
       ts=arrays(np.float, 10, elements = floats  (min_value =    2*mus, max_value = 100*mus)),
       ps=arrays(np.int32, 10, elements = integers(min_value =    10   , max_value = 100    )))
def test_create_wfs_pmts_shape(get_dfs, xs, ys, ts, ps):
    fname, lt_df, lt_conf = get_dfs['lt']
    el_drift_velocity = 2.5 * mm/mus
    sensor_time_bin   = 100 * ns
    buffer_length     = 200 * mus
    lt = LT_PMT(fname=fname)
    n_sensors = lt.num_sensors
    waveform = create_wfs(xs, ys, ts, ps, lt, el_drift_velocity, sensor_time_bin, buffer_length)
    assert isinstance(waveform, np.ndarray)
    assert waveform.shape == (n_sensors, buffer_length//sensor_time_bin)


@given(xs=arrays(np.float, 10, elements = floats  (min_value = -500*mm , max_value = 500*mm )),
       ys=arrays(np.float, 10, elements = floats  (min_value = -500*mm , max_value = 500*mm )),
       ts=arrays(np.float, 10, elements = floats  (min_value =    2*mus, max_value = 200*mus)),
       ps=arrays(np.int32, 10, elements = integers(min_value =    10   , max_value = 100    )))
def test_create_wfs_tmin(get_dfs, xs, ys, ts, ps):
    datasipm = DataSiPM('new')
    fname, psf_df, psf_conf = get_dfs['psf']
    el_drift_velocity = 2.5 * mm/mus
    sensor_time_bin   = 1   * mus
    buffer_length     = 100 * mus
    lt = LT_SiPM(fname=fname, sipm_database=datasipm)
    n_sensors = lt.num_sensors
    tmin = 100*mus
    ts_shift = ts - tmin
    waveform_sh = create_wfs(xs, ys, ts_shift, ps, lt, el_drift_velocity, sensor_time_bin, buffer_length)
    waveform    = create_wfs(xs, ys, ts      , ps, lt, el_drift_velocity, sensor_time_bin, buffer_length, tmin)
    np.testing.assert_allclose(waveform_sh, waveform)


@given(xs=arrays(np.float, 10, elements = floats  (min_value = -500*mm , max_value = 500*mm )),
       ys=arrays(np.float, 10, elements = floats  (min_value = -500*mm , max_value = 500*mm )),
       ts=arrays(np.float, 10, elements = floats  (min_value =    2*mus, max_value = 100*mus)),
       ps=arrays(np.int32, 10, elements = integers(min_value =    10   , max_value = 100    )))
def test_integrated_signal_pmts(get_dfs, xs, ys, ts, ps):
    fname, lt_df, lt_conf = get_dfs['lt']
    el_drift_velocity = 2.5 * mm/mus
    sensor_time_bin   = 100 * ns
    buffer_length     = 200 * mus
    lt = LT_PMT(fname=fname)
    n_sensors = lt.num_sensors
    waveform = create_wfs(xs, ys, ts, ps, lt, el_drift_velocity, sensor_time_bin, buffer_length)
    #calculate integrated signal from light tables per pmt
    for i in range(12): #12 pmts
        summed_sig  = np.sum([lt.get_values(x, y, i)*p for x, y, p in zip(xs, ys, ps)])
        assert np.isclose(waveform[i].sum(),summed_sig)


@given(xs=arrays(np.float, 10, elements = floats  (min_value = -500*mm , max_value = 500*mm )),
       ys=arrays(np.float, 10, elements = floats  (min_value = -500*mm , max_value = 500*mm )),
       ts=arrays(np.float, 10, elements = floats  (min_value =    2*mus, max_value = 100*mus)),
       ps=arrays(np.int32, 10, elements = integers(min_value =    10   , max_value = 100    )))
def test_integrated_signal_sipms(get_dfs, xs, ys, ts, ps):
    datasipm = DataSiPM('new')
    fname, psf_df, psf_conf = get_dfs['psf']
    el_drift_velocity = 2.5 * mm/mus
    sensor_time_bin   = 1   * mus
    buffer_length     = 200 * mus
    lt = LT_SiPM(fname=fname, sipm_database=datasipm)
    n_sensors = lt.num_sensors
    waveform = create_wfs(xs, ys, ts, ps, lt, el_drift_velocity, sensor_time_bin, buffer_length)
    #calculate integrated signal from light tables per pmt
    for i in range(len(datasipm)):
        summed_sig  = np.sum([lt.get_values(x, y, i)*p for x, y, p in zip(xs, ys, ps)])
        assert np.isclose(waveform[i].sum(),summed_sig)


@given(xs=floats  (min_value = -500*mm , max_value = 500*mm ),
       ys=floats  (min_value = -500*mm , max_value = 500*mm ),
       ts=floats  (min_value =    2*mus, max_value = 10 *mus),
       ps=integers(min_value =   10    , max_value = 100   ))
def test_time_distribution_pmts(get_dfs, xs, ys, ts, ps):
    fname, lt_df, lt_conf = get_dfs['lt']
    lt = LT_PMT(fname=fname)
    el_drift_velocity = 2.5 * mm/mus
    sensor_time_bin   = 10  * ns
    buffer_length     = 20  * mus
    time_bins = np.arange(0, buffer_length, sensor_time_bin)
    el_time   = lt.el_gap_width/el_drift_velocity
    tindx_min = np.digitize(ts, time_bins)-1
    tindx_max = np.digitize(ts+el_time, time_bins)-1
    nbins_el_gap = tindx_max-tindx_min
    n_sensors = lt.num_sensors
    waveform = create_wfs(np.array([xs]), np.array([ys]), np.array([ts]), np.array([ps]).astype(np.intc), lt, el_drift_velocity, sensor_time_bin, buffer_length)
    for i in range(12): #12 pmts
        total = lt.get_values(xs, ys, i)*ps
        #the signal is expected to be uniformely spread
        expected_wf = np.ones(shape=nbins_el_gap)*total/nbins_el_gap
        np.testing.assert_allclose(waveform[i, tindx_min:tindx_max], expected_wf)


@given(xs=floats  (min_value = -10*mm , max_value = 10 *mm ),
       ys=floats  (min_value = -10*mm , max_value = 10 *mm ),
       ts=floats  (min_value =   2*mus, max_value = 10 *mus),
       ps=integers(min_value =  2000  , max_value = 3000   ))
def test_time_distribution_sipms(get_dfs, xs, ys, ts, ps):
    datasipm = DataSiPM('new')
    #select central sipms only that will see the signal
    datasipm = datasipm[(np.abs(datasipm.X)<=10*mm)&(np.abs(datasipm.Y)<=10*mm)]
    fname, psf_df, psf_conf = get_dfs['psf']
    el_drift_velocity = 2.5 * mm/mus
    sensor_time_bin   = 1   * mus
    buffer_length     = 20  * mus
    time_bins = np.arange(0, buffer_length+sensor_time_bin, sensor_time_bin)
    lt = LT_SiPM(fname=fname, sipm_database=datasipm)
    el_time   = lt.el_gap_width/el_drift_velocity
    tindx_min = np.digitize(ts, time_bins)-1
    tindx_max = np.digitize(ts+el_time, time_bins)-1
    nbins_el_gap = tindx_max-tindx_min
    n_sensors = lt.num_sensors
    waveform = create_wfs(np.array([xs]), np.array([ys]), np.array([ts]), np.array([ps]).astype(np.intc), lt, el_drift_velocity, sensor_time_bin, buffer_length)
    z_bins = lt.zbins
    z_bins_time = ts + z_bins/el_drift_velocity
    for i in range(len(datasipm)):
        signal = lt.get_values(xs, ys, i)*ps
        expected, _ = np.histogram(z_bins_time, bins=time_bins, weights=signal)
        np.testing.assert_allclose(waveform[i], expected)

