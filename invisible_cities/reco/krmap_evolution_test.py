import numpy                                as np
import pandas                               as pd
import numpy.testing                        as npt

from   pytest                               import approx, mark
from   flaky                                import flaky

from   hypothesis                           import given
from   hypothesis.strategies                import floats, integers
from .. reco.krmap_evolution                import sigmoid, gauss_seed, compute_drift_v, resolution
from .. reco.krmap_evolution                import quick_gauss_fit, get_time_series_df

from .. evm.ic_containers                   import FitFunction
from .. database                            import load_db  as DB
from .. io.dst_io                           import load_dst
from .. core.testing_utils                  import assert_dataframes_equal


def test_sigmoid_definition_fixed_values():

    x          = np.array(range(10))
    scale      =  1
    slope      = -5
    offset     =  0
    inflection =  5

    expected_output = scale / (1 + np.exp(-slope * (x - inflection))) + offset
    actual_output   = sigmoid(x, scale, inflection, slope, offset)

    npt.assert_allclose(actual_output, expected_output, rtol=1e-10)


def test_sigmoid_zero_slope():

    x          = np.array(range(10))
    scale      = 1
    slope      = 0
    offset     = 0
    inflection = 0

    expected_output = np.full(len(x), 0.5 * scale + offset)
    actual_output   = sigmoid(x, scale, inflection, slope, offset)

    npt.assert_allclose(actual_output, expected_output, rtol=1e-10)


@given(floats(min_value = 1e-2,  max_value = 1e4),
       floats(min_value = -1e2,  max_value = 1e2))
def test_sigmoid_limits(scale, offset):

    slope       = 10
    inflection  = 0
    aux_sigmoid = lambda x: sigmoid(x, scale, inflection, slope, offset)

    assert aux_sigmoid(-10) == approx(offset,       rel=1e-2)
    assert aux_sigmoid( 10) == approx(scale+offset, rel=1e-2)


@given(floats(min_value = 1e-2,  max_value = 1e4),
       floats(min_value = -10,   max_value = 10),
       floats(min_value = 1e-2,  max_value = 1e2),
       floats(min_value = -1e2,  max_value = 1e2))
def test_sigmoid_values_at_0(scale, inflection, slope, offset):
    expected_output = scale / (1 + np.exp(slope*inflection)) + offset
    actual_output   = sigmoid(0, scale, inflection, slope, offset)
    assert expected_output == actual_output


@given(floats(min_value = 1e-3, max_value = 1))
def test_gauss_seed_all_sigmas(sigma_rel):

    x = np.array(range(10))
    y = np.array([1, 3, 7, 12, 18, 25, 18, 12, 7, 3])

    max_y = np.max(y)
    max_x = x[np.argmax(y)]

    expected_amp  = max_y * (2 * np.pi)**0.5 * (sigma_rel * (max(x)-min(x))*0.5)
    expected_seed = (expected_amp, max_x, sigma_rel * (max(x)-min(x))*0.5)

    actual_seed = gauss_seed(x, y, sigma_rel=sigma_rel)
    npt.assert_allclose(actual_seed, expected_seed, rtol=1e-5)


@given(floats  (min_value=-1e2, max_value=1e4),
       floats  (min_value=0.1,  max_value=5),
       integers(min_value=1e2,  max_value=1e5),
       integers(min_value=10,   max_value = 1e2))
def test_quick_gauss_fit(mean, sigma, size, bins):

    def generate_gaussian_data(mean, sigma, size):
        return np.random.normal(loc=mean, scale=sigma, size=size)

    data       = generate_gaussian_data(mean, sigma, size)
    fit_result = quick_gauss_fit(data, bins)

    assert type(fit_result) is FitFunction
    assert np.isclose(fit_result.values[1], mean,  rtol=0.2)
    assert np.isclose(fit_result.values[2], sigma, rtol=0.2)


@flaky(max_runs=10, min_passes=9)
def test_compute_drift_v_when_moving_edge():
    edge    = np.random.uniform(530, 570)
    Nevents = 100 * 1000
    data    = np.random.uniform(450, edge, Nevents)
    data    = np.random.normal(data, 1)
    dv, dvu = compute_drift_v(dtdata = data, nbins = 60, dtrange = [500, 600],
                              seed = [1500, 550, 1, 0], detector ='new')
    dv_th   = DB.DetectorGeo('new').ZMAX[0]/edge

    assert dv_th == approx(dv, abs=5*dvu)


def test_compute_drift_v_failing_fit_return_nan():
    dst     = np.random.rand(1000)*1200
    dv_vect = compute_drift_v(dtdata = dst, nbins = 50, dtrange = [500, 600],
                              seed = [1500, 550, 1, 0], detector ='new')

    assert np.all(np.isnan(dv_vect))


def test_get_time_series_df_empty_data():

    dst         = pd.DataFrame({'time': []})
    ntimebins   = 3
    time_range  = (0, 10)
    ts, masks   = get_time_series_df(ntimebins, time_range, dst)
    expected_ts = np.linspace(0, 10, ntimebins + 1)[:-1] + (10 / (2 * ntimebins))

    npt.assert_allclose(ts, expected_ts, rtol=1e-5)

    for mask in masks:
        assert np.sum(mask) == 0


@mark.parametrize('n', [2, 5, 10])
def test_get_time_series_df_pandas(KrMC_kdst, n):
    kdst = load_dst(KrMC_kdst, 'DST', 'Events')
    time_indx = pd.cut(kdst.time, n, labels=np.arange(n))
    ts, masks = get_time_series_df(n, (kdst.time.min(), kdst.time.max()), kdst)
    for i, mask in enumerate(masks):
        assert_dataframes_equal(kdst[mask], kdst[time_indx==i])


def test_resolution_definition():

    values = np.array([10, 100, 2])
    errors = np.array([0.1, 1, 0.05])
    expected_res  = 235.48 * (values[2] / values[1])  # 235.48 * sigma/mu
    expected_ures = expected_res * ((errors[1] / values[1])**2 + (errors[2] / values[2])**2)**0.5

    res, ures = resolution(values, errors)

    assert res  == approx(expected_res,  rel=1e-5)
    assert ures == approx(expected_ures, rel=1e-5)