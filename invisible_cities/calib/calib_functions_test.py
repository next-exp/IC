import os

import numpy  as np
import tables as tb

from numpy.testing import assert_array_equal
from numpy.testing import assert_approx_equal
from pytest        import mark
from pytest        import raises
from scipy.signal  import find_peaks_cwt

from .                         import calib_functions as cf
from .. reco                   import tbl_functions   as tbl
from .. core                   import fit_functions   as fitf
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


def test_seeds_db():
    gain_seed_sipm, gain_sigma_seed_sipm = seeds_db('sipm', 6317, 13)
    gain_seed_pmt, gain_sigma_seed_pmt   = seeds_db('pmt', 6317, 5)

    assert gain_seed_sipm       == 16.5503
    assert gain_sigma_seed_sipm == 1.65978
    assert gain_seed_pmt        == 22.66
    assert gain_sigma_seed_pmt  == 9.88


def test_poisson_mu_seed():
    bins     = np.array([-8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7])
    spec     = np.array([28, 98, 28, 539, 1072, 1845, 2805, 3251, 3626, 3532, 3097, 2172, 1299, 665, 371, 174])
    dark     = np.array([30, 107, 258, 612, 1142, 2054, 3037, 3593, 3769, 3777, 3319, 2321, 1298, 690, 415, 192])
    ped_vals = np.array([2.65181178e+04, 1.23743445e-01, 2.63794236e+00])

    scaler   = dark_scaler(dark[(bins>=-5) & (bins<=5)])
    scaler2  = dark_scaler(dark[bins<0])
    mu       = poisson_mu_seed('sipm', bins, spec, ped_vals, scaler)
    mu2      = poisson_mu_seed('pmt', bins, spec, ped_vals, scaler2)

    np.testing.assert_approx_equal(mu, 0.0698154)
    np.testing.assert_approx_equal(mu2, 0.0950066)


def test_sensor_values():
    bins     = np.array([-6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7])
    spec     = np.array([28, 539, 1072, 1845, 2805, 3251, 3626, 3532, 3097, 2172, 1299, 665, 371, 174])
    dark     = np.array([258, 612, 1142, 2054, 3037, 3593, 3769, 3777, 3319, 2321, 1298, 690, 415, 192])
    ped_vals = np.array([2.65181178e+04, 1.23743445e-01, 2.63794236e+00])
    scaler   = dark_scaler(dark[(bins>=-5) & (bins<=5)])
    scaler2  = dark_scaler(dark[bins<0])
    spectra, p_range, min_bin, max_bin, hpw, lim_ped = sensor_values('sipm', 1023, scaler_func, spec, bins, ped_vals)
    spectra2, p_range2, min_bin2, max_bin2, hpw2, lim_ped2 = sensor_values('pmt', 0, scaler_func, spec, bins, ped_vals)

    expected_range = np.arange(4,20)
    expected_spec2 = np.array([-215.61455106, -7.61389796, 9.69755144, 56.84252052, 197.9256732, -41.23729614, 25.03486623,
                               120.56759537, 297.7306255, 182.5057727, 74.29711143, 12.00648349, 69.4376878, 53.375555])
    expected_range2 = np.arange(10,20)

    np.testing.assert_array_equal(spectra, spec)
    np.testing.assert_array_equal(p_range, expected_range)
    assert min_bin == 10
    assert max_bin == 22
    assert hpw == 5
    assert lim_ped == lim_ped2
    assert len(spectra2) == len(expected_spec2)

    np.testing.assert_array_equal(p_range2, expected_range2)
    assert min_bin2 == 15
    assert max_bin2 == 50
    assert hpw2 == 10


def test_pedestal_values():
    ped_vals = np.array([6.14871401e+04, -1.46181517e-01, 5.27614635e+00])
    ped_errs = np.array([9.88752708e+02, 5.38541961e-02, 1.07169703e-01])
    lim_ped  = 10000

    p_seed, p_sig_seed, p_min, p_max, p_sig_min, p_sig_max = pedestal_values(ped_vals,
                                                                             lim_ped, ped_errs)

    assert_approx_equal(p_seed, -0.14618, 5)
    assert_approx_equal(p_sig_seed, 5.27614, 5)
    assert_approx_equal(p_min, -538.68814, 5)
    assert_approx_equal(p_max, 538.39577, 5)
    assert_approx_equal(p_sig_min, 0.001)
    assert_approx_equal(p_sig_max, 1076.97317, 5)


def test_seeds_and_bounds(file_name):
    sipmIn = tb.open_file(file_name, 'r')
    run_no = file_name[file_name.find('R')+1:file_name.find('R')+5]
    run_no = int(run_no)

    specsL = np.array(sipmIn.root.HIST.sipm_spe).sum(axis=0)
    specsD = np.array(sipmIn.root.HIST.sipm_dark).sum(axis=0)
    bins   = np.array(sipmIn.root.HIST.sipm_spe_bins)

    min_stat = 10

    for ich, (led, dar) in enumerate(zip(specsL, specsD)):
        valid_bins = np.argwhere(led>=min_stat)
        b1 = valid_bins[0][0]
        b2 = valid_bins[-1][0]
        pD = find_peaks_cwt(dar, np.arange(2, 20), min_snr=2)
        if len(pD) == 0:
            print('no peaks in dark spectrum, spec ', ich)
            continue

        gb0     = [(0, -100, 0), (1e99, 100, 10000)]
        sd0     = (dar.sum(), 0, 2)
        errs    = poisson_sigma(dar[pD[0]-5:pD[0]+5], default=0.1)
        gfitRes = fitf.fit(fitf.gauss, bins[pD[0]-5:pD[0]+5], dar[pD[0]-5:pD[0]+5], sd0, sigma=errs, bounds=gb0)

        ped_vals    = np.array([gfitRes.values[0], gfitRes.values[1], gfitRes.values[2]])
        scaler_func = dark_scaler(dar[b1:b2][(bins[b1:b2]>=-5) & (bins[b1:b2]<=5)])

        seeds, bounds = seeds_and_bounds('sipm', run_no, ich, scaler_func, bins[b1:b2], led[b1:b2],
                                         ped_vals, gfitRes.errors, use_db_gain_seeds=False)

        assert all(i > 0 for i in seeds)
        assert bounds[0] == (0, 0, 0, 0.001)
        assert bounds[1] == (10000000000.0, 10000, 10000, 10000)

