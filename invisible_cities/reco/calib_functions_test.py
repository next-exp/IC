import os

import numpy  as np
import tables as tb

from numpy.testing import assert_array_equal
from numpy.testing import assert_approx_equal
from pytest        import mark
from pytest        import raises
from scipy.signal  import find_peaks_cwt

from .                           import calib_functions as cf
from .. reco                     import tbl_functions   as tbl
from .. core                     import fit_functions   as fitf
from .. core.stat_functions      import poisson_sigma
from .. core.system_of_units_c   import units
from .. evm.nh5                  import SensorTable
from .  calib_functions          import SensorType
from .. liquid_cities.components import get_run_number


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


@mark.parametrize('sensor_type     , n_channel, gain_seed, gain_sigma_seed',
                  ((SensorType.SIPM,         1,   16.5622,         2.5),
                   (SensorType.PMT ,         5,   24.9557,         9.55162)))
def test_seeds_db(sensor_type, n_channel, gain_seed, gain_sigma_seed):
    run_number = 6217
    result = cf.seeds_db(sensor_type, run_number, n_channel)
    assert result == (gain_seed, gain_sigma_seed)


dark_sipm = np.array([612, 1142, 2054, 3037, 3593, 3769, 3777, 3319, 2321, 1298, 690])
dark_pmt  = np.array([ 30,  107,  258,  612, 1142, 2054, 3037, 3593                 ])

@mark.parametrize('     sensor_type,                    scaler,        mu',
                  ((SensorType.SIPM, cf.dark_scaler(dark_sipm), 0.0698154),
                   (SensorType.PMT , cf.dark_scaler(dark_pmt) , 0.0950066)))
def test_poisson_mu_seed(sensor_type, scaler, mu):
    bins     = np.array([-8,  -7,  -6,  -5,  -4,    -3,   -2,   -1,    0,    1,    2,    3,    4,   5,   6,   7])
    spec     = np.array([28,  98,  28, 539, 1072, 1845, 2805, 3251, 3626, 3532, 3097, 2172, 1299, 665, 371, 174])
    ped_vals = np.array([2.65181178e+04, 1.23743445e-01, 2.63794236e+00])

    result   = cf.poisson_mu_seed(sensor_type, scaler, bins, spec, ped_vals)
    assert_approx_equal(result, mu)


@mark.parametrize('     sensor_type, n_chann,                    scaler,    expected_range, min_b, max_b, half_width, p1pe_seed, lim_p',
                  ((SensorType.SIPM,    1023, cf.dark_scaler(dark_sipm),  np.arange(4 ,20),    10,    22,          5,         3, 10000),
                   (SensorType.PMT ,       0, cf.dark_scaler(dark_pmt) ,  np.arange(10,20),    15,    50,         10,         7, 10000)))
def test_sensor_values(sensor_type, n_chann, scaler, expected_range, min_b, max_b, half_width, p1pe_seed, lim_p):
    bins     = np.array([ -6,  -5,   -4,   -3,   -2,   -1,    0,    1,    2,    3,    4,   5,   6,   7])
    spec     = np.array([ 28, 539, 1072, 1845, 2805, 3251, 3626, 3532, 3097, 2172, 1299, 665, 371, 174])
    ped_vals = np.array([2.65181178e+04, 1.23743445e-01, 2.63794236e+00])
    spectra, p_range, min_bin, max_bin, hpw, seed, lim_ped = cf.sensor_values(sensor_type, n_chann, scaler, bins, spec, ped_vals)

    assert_array_equal(p_range, expected_range)
    assert len(spectra) == len(spec)
    assert min_bin      == min_b
    assert max_bin      == max_b
    assert hpw          == half_width
    assert seed         == p1pe_seed
    assert lim_ped      == lim_p


@mark.parametrize('sensor_type, run_number, n_chann,                    scaler',
                  ((      None,       6217,    1023, cf.dark_scaler(dark_sipm)),
                   (      None,       6217,       0, cf.dark_scaler(dark_pmt))))
def test_incorrect_sensor_type_raises_ValueError(sensor_type, run_number, n_chann, scaler):
    bins     = np.array([ -6,  -5,   -4,   -3,   -2,   -1,    0,    1,    2,    3,    4,   5,   6,   7])
    spec     = np.array([ 28, 539, 1072, 1845, 2805, 3251, 3626, 3532, 3097, 2172, 1299, 665, 371, 174])
    ped_vals = np.array([2.65181178e+04, 1.23743445e-01, 2.63794236e+00])

    with raises(ValueError):
        cf.       seeds_db(sensor_type, run_number, n_chann)
        cf.poisson_mu_seed(sensor_type, scaler, bins, spec, ped_vals)
        cf.  sensor_values(sensor_type, n_chann, scaler, bins, spec, ped_vals)


def test_pedestal_values():
    ped_vals   = np.array([6.14871401e+04, -1.46181517e-01, 5.27614635e+00])
    ped_errs   = np.array([9.88752708e+02,  5.38541961e-02, 1.07169703e-01])
    ped_values = cf.pedestal_values(ped_vals, 10000, ped_errs)

    assert_approx_equal(ped_values.gain     ,   -0.14618, 5)
    assert_approx_equal(ped_values.sigma    ,    5.27614, 5)
    assert_approx_equal(ped_values.gain_min , -538.68814, 5)
    assert_approx_equal(ped_values.gain_max ,  538.39577, 5)
    assert_approx_equal(ped_values.sigma_max, 1076.97317, 5)
    assert_approx_equal(ped_values.sigma_min,      0.001)


def test_seeds_without_using_db(ICDATADIR):
    PATH_IN = os.path.join(ICDATADIR, 'sipmcalspectra_R6358.h5')
    h5in    = tb.open_file(PATH_IN, 'r')
    run_no  = get_run_number(h5in)

    specsL = np.array(h5in.root.HIST.sipm_spe).sum(axis=0)
    specsD = np.array(h5in.root.HIST.sipm_dark).sum(axis=0)
    bins   = np.array(h5in.root.HIST.sipm_spe_bins)

    min_stat = 10

    for ich, (led, dar) in enumerate(zip(specsL, specsD)):
        b1 = 0
        b2 = len(dar)
        if min_stat != 0:
            try:
                valid_bins = np.argwhere(led>=min_stat)
                b1 = valid_bins[ 0][0]
                b2 = valid_bins[-1][0]
            except IndexError:
                continue

        pD = find_peaks_cwt(dar, np.arange(2, 20), min_snr=2)
        if len(pD) == 0:
            continue

        gb0     = [(0, -100, 0), (np.inf, 100, 10000)]
        sd0     = (dar.sum(), 0, 2)
        sel     = np.arange(pD[0]-5, pD[0]+5)
        errs    = poisson_sigma(dar[sel], default=0.1)
        gfitRes = fitf.fit(fitf.gauss, bins[sel], dar[sel], sd0, sigma=errs, bounds=gb0)

        ped_vals      = np.array([gfitRes.values[0], gfitRes.values[1], gfitRes.values[2]])
        scaler_func   = cf.dark_scaler(dar[b1:b2][(bins[b1:b2]>=-5) & (bins[b1:b2]<=5)])
        seeds, bounds = cf.seeds_and_bounds(SensorType.SIPM, run_no, ich, scaler_func, bins[b1:b2],
                                            led[b1:b2], ped_vals, gfitRes.errors, use_db_gain_seeds=False)
        assert not (all(seeds) == 0)
        assert bounds == ((0, 0, 0, 0.001), (1e10, 10000, 10000, 10000))
