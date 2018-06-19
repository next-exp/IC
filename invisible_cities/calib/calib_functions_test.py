import os

import numpy  as np
import tables as tb

from numpy.testing import assert_array_equal
from pytest        import mark
from pytest        import raises

from .                         import calib_functions as cf
from .. reco                   import tbl_functions as tbl
from .. core.system_of_units_c import units
from .. evm.nh5                import SensorTable


def test_bin_waveforms():
    bins = np.arange(0, 20)
    data = np.arange(1, 51).reshape(2, 25)

    expected = np.stack((np.histogram(data[0], bins)[0],
                         np.histogram(data[1], bins)[0]))
    actual   = cf.bin_waveforms(data, bins)
    assert_array_equal(actual, expected)


def test_spaced_integrals():
    limits = np.array([2, 4, 6])
    data   = np.arange(20).reshape(2, 10)

    expected = np.array([[5, 9, 30], [25, 29, 70]])
    actual   = cf.spaced_integrals(data, limits)
    assert_array_equal(actual, expected)


@mark.parametrize("limits",
                  ([-1, 0,  2],
                   [-1, 1, 10],
                   [ 0, 2, 10],
                   [ 0, 3, 11]))
def test_spaced_integrals_raises_ValueError_with_wrong_limits(limits):
    limits = np.array(limits)
    data   = np.arange(20).reshape(2, 10)
    with raises(ValueError):
        cf.spaced_integrals(data, limits)


def test_integral_limits():
    sampling    =  1 * units.mus
    n_integrals = 10
    start_int   =  5
    width_int   =  1
    period      = 50

    expected_llim = np.array([  5,   6,  55,  56, 105, 106, 155, 156, 205, 206, 255, 256, 305, 306, 355, 356, 405, 406, 455, 456])
    expected_dlim = np.array([  2,   3,  52,  53, 102, 103, 152, 153, 202, 203, 252, 253, 302, 303, 352, 353, 402, 403, 452, 453])

    (actual_llimits,
     actual_dlimits) = cf.integral_limits(sampling, n_integrals, start_int, width_int, period)

    assert_array_equal(actual_llimits, expected_llim)
    assert_array_equal(actual_dlimits, expected_dlim)


def test_filter_limits_inside():
    sampling         =   1 * units.mus
    n_integrals      =  10
    start_int        =   5
    width_int        =   1
    period           =  50
    fake_data_length = 500

    (expected_llimits,
     expected_dlimits) = cf.integral_limits(sampling, n_integrals, start_int, width_int, period)

    actual_llimits = cf.filter_limits(expected_llimits, fake_data_length)
    actual_dlimits = cf.filter_limits(expected_dlimits, fake_data_length)

    assert_array_equal(actual_llimits, expected_llimits)
    assert_array_equal(actual_dlimits, expected_dlimits)


def test_filter_limits_outside():
    sampling         =   1 * units.mus
    n_integrals      =  10
    start_int        =   5
    width_int        =   1
    period           =  50
    fake_data_length = 400

    (unfiltered_llimits,
     unfiltered_dlimits) = cf.integral_limits(sampling, n_integrals, start_int, width_int, period)

    filtered_llimits = cf.filter_limits(unfiltered_llimits, fake_data_length)
    filtered_dlimits = cf.filter_limits(unfiltered_dlimits, fake_data_length)

    assert len(filtered_llimits) < len(unfiltered_llimits)
    assert len(filtered_dlimits) < len(unfiltered_dlimits)
    assert len(filtered_llimits) % 2 == 0
    assert len(filtered_dlimits) % 2 == 0


def test_copy_sensor_table(config_tmpdir):

    ## Create an empty input file to begin
    in_name  = os.path.join(config_tmpdir, 'test_copy_in.h5')
    out_name = os.path.join(config_tmpdir, 'test_copy_out.h5')
    with tb.open_file(in_name, 'w') as input_file:
        input_file.create_group(input_file.root, 'dummy')
    
    ## Test copy where Sensors group not present etc.
    with tb.open_file(out_name, 'w') as out_file:

        ## Nothing to copy
        cf.copy_sensor_table(in_name, out_file)

        ## Sensor group with no table
        with tb.open_file(in_name, 'a') as input_file:
            sens_group = input_file.create_group(input_file.root, 'Sensors')
        cf.copy_sensor_table(in_name, out_file)
        assert 'Sensors' in out_file.root


def test_copy_sensor_table2(config_tmpdir):

    ## Create an empty input file to begin
    in_name  = os.path.join(config_tmpdir, 'test_copy_in.h5')
    out_name = os.path.join(config_tmpdir, 'test_copy_out.h5')
    ## Only PMTs
    dummy_pmt = (0, 11)
    with tb.open_file(in_name, 'w') as input_file:
        sens_group = input_file.create_group(input_file.root, 'Sensors')
        pmt_table  = input_file.create_table(sens_group, "DataPMT", SensorTable,
                                             "", tbl.filters("NOCOMPR"))
        row = pmt_table.row
        row["channel"]  = dummy_pmt[0]
        row["sensorID"] = dummy_pmt[1]
        row.append()
        pmt_table.flush

    ## Test copy where Sensors group not present etc.
    with tb.open_file(out_name, 'w') as out_file:
        cf.copy_sensor_table(in_name, out_file)
        assert 'DataPMT' in out_file.root.Sensors
        assert out_file.root.Sensors.DataPMT[0][0] == dummy_pmt[0]
        assert out_file.root.Sensors.DataPMT[0][1] == dummy_pmt[1]


def test_copy_sensor_table3(config_tmpdir):

    ## Create an empty input file to begin
    in_name  = os.path.join(config_tmpdir, 'test_copy_in.h5')
    out_name = os.path.join(config_tmpdir, 'test_copy_out.h5')
    ## Only SiPMs
    dummy_sipm = (1013, 1000)
    with tb.open_file(in_name, 'w') as input_file:
        sens_group = input_file.create_group(input_file.root, 'Sensors')
        sipm_table = input_file.create_table(sens_group, "DataSiPM", SensorTable,
                                             "", tbl.filters("NOCOMPR"))
        row = sipm_table.row
        row["channel"]  = dummy_sipm[0]
        row["sensorID"] = dummy_sipm[1]
        row.append()
        sipm_table.flush

    ## Test copy where Sensors group not present etc.
    with tb.open_file(out_name, 'w') as out_file:
        cf.copy_sensor_table(in_name, out_file)
        assert 'DataSiPM' in out_file.root.Sensors
        assert out_file.root.Sensors.DataSiPM[0][0] == dummy_sipm[0]
        assert out_file.root.Sensors.DataSiPM[0][1] == dummy_sipm[1]
        
