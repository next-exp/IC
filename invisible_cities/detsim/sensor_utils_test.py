import numpy  as np
import pandas as pd

from pytest import fixture
from pytest import    mark

from .. core import system_of_units as units

from . sensor_utils import   first_and_last_times
from . sensor_utils import          get_n_sensors
from . sensor_utils import           sensor_order
from . sensor_utils import pmt_and_sipm_bin_width
from . sensor_utils import          trigger_times
from . sensor_utils import       create_timestamp


def test_trigger_times():
    event_time = 22 * units.ns
    bin_width  = 0.1
    time_bins  = np.arange(0, 10, bin_width)
    triggers   = [15, 75]

    trgr_times = trigger_times(triggers, event_time, time_bins)
    assert len(trgr_times) == len(triggers)
    assert trgr_times[0]   == event_time + triggers[0] * bin_width
    assert trgr_times[1]   == event_time + triggers[1] * bin_width


@mark.parametrize("bin_min bin_max bin_wid".split(),
                  ((22, 40, 1), (1, 5, 0.1)))
def test_first_and_last_times(bin_min, bin_max, bin_wid):
    test_bins1 = np.arange(bin_min, bin_max, bin_wid)
    test_bins2 = np.arange(bin_min, bin_max - 1)

    bmin, bmax = first_and_last_times(pd.DataFrame({'time': test_bins1}),
                                      pd.DataFrame({'time': test_bins2}),
                                      bin_wid, 1)

    assert np.isclose(bmin, bin_min)
    assert np.isclose(bmax, bin_max)


@fixture(scope = 'function')
def sensor_info():
    id_dict    = {'pmt_ids' :    (5, 2, 7), 'pmt_ord' :  (5, 2, 7),
                  'sipm_ids': (1010, 5023), 'sipm_ord':  (10, 279)}

    nsamp_pmt  = 5000
    nsamp_sipm =    5
    pmt_resp   = pd.Series([[1]*nsamp_pmt ]*len(id_dict[ 'pmt_ids']),
                           index = id_dict[ 'pmt_ids'])
    sipm_resp  = pd.Series([[1]*nsamp_sipm]*len(id_dict['sipm_ids']),
                           index = id_dict['sipm_ids'])
    pmt_q      = np.array( pmt_resp.tolist())
    sipm_q     = np.array(sipm_resp.tolist())
    return id_dict, nsamp_pmt, nsamp_sipm, pmt_resp, sipm_resp, pmt_q, sipm_q


def test_sensor_order(sensor_info, pmt_ids, sipm_ids):
    detector_db   = 'new'
    run_number    = 6400
    n_pmt         = len( pmt_ids)
    n_sipm        = len(sipm_ids)

    (id_dict , nsamp_pmt, nsamp_sipm,
     pmt_resp, sipm_resp, pmt_q     , sipm_q) = sensor_info

    order_sensors = sensor_order(detector_db, run_number,
                                 nsamp_pmt  , nsamp_sipm)

    padded_evt = order_sensors(pmt_resp, sipm_resp, [(pmt_q, sipm_q)])

    pmt_out  = padded_evt[0][0]
    sipm_out = padded_evt[0][1]
    assert  pmt_out.shape == (n_pmt ,  nsamp_pmt)
    assert sipm_out.shape == (n_sipm, nsamp_sipm)

    pmt_nonzero  = np.argwhere( pmt_out.sum(axis=1) != 0)
    sipm_nonzero = np.argwhere(sipm_out.sum(axis=1) != 0)

    assert np.all([pmt  in id_dict[ 'pmt_ord'] for pmt  in  pmt_nonzero])
    assert np.all([sipm in id_dict['sipm_ord'] for sipm in sipm_nonzero])


def test_get_n_sensors(pmt_ids, sipm_ids):
    npmt, nsipm = get_n_sensors('new', -6400)

    assert npmt  == len( pmt_ids)
    assert nsipm == len(sipm_ids)


def test_pmt_and_sipm_bin_width(full_sim_file):
    file_in = full_sim_file

    expected_pmtwid  = 100 * units.ns
    expected_sipmwid =   1 * units.mus

    pmt_binwid, sipm_binwid = pmt_and_sipm_bin_width(file_in)
    assert pmt_binwid  == expected_pmtwid
    assert sipm_binwid == expected_sipmwid


def test_create_timestamp_greater_with_greater_arguments():
    """
    Value of timestamp must be always positive and 
    greater with greater arguments.
    """
    evt_no_1 = 10
    rate_1   = 0.5
    evt_no_2 = 100
    rate_2   = 0.6

    timestamp_1 = create_timestamp(evt_no_1, rate_1)
    timestamp_2 = create_timestamp(evt_no_2, rate_2)
    assert abs(timestamp_1) == timestamp_1
    assert timestamp_1      <  timestamp_2
    assert abs(timestamp_2) == timestamp_2
