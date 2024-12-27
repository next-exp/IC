import numpy                                as np
import pandas                               as pd
import numpy.testing                        as npt

from   pytest                               import approx, mark, fixture
from   flaky                                import flaky

from   hypothesis                           import given
from   hypothesis.strategies                import floats, integers
from .. reco.krmap_evolution                import sigmoid, gauss_seed, compute_drift_v, resolution
from .. reco.krmap_evolution                import quick_gauss_fit, get_time_series_df, computing_kr_parameters

from .. evm.ic_containers                   import FitFunction
from .. types.symbols                       import KrFitFunction
from .. database                            import load_db  as DB


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


@mark.parametrize('mean sigma size bins'.split(),
(  (0, 1,  1000,   50),
  (10, 2, 10000,  100),
(-100, 1,  5000,   75),)
)
def test_quick_gauss_fit(mean, sigma, size, bins):

    def generate_gaussian_data(mean, sigma, size):
        return np.random.normal(loc=mean, scale=sigma, size=size)

    data = generate_gaussian_data(mean, sigma, size)

    fit_result = quick_gauss_fit(data, bins)

    print(fit_result.values[1])
    print(mean)

    assert type(fit_result) is FitFunction
    assert np.isclose(fit_result.values[2], sigma, rtol=0.1)
    assert np.isclose(fit_result.values[1], mean,  atol=0.1)


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
    '''Generating nonsense data to make sigmoid fit fail'''
    dst = [500, 599]
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

@mark.parametrize('reverse', [False, True])
@mark.parametrize('n', [2, 5, 10])
def test_get_time_series_df_pandas(n, reverse):

    time = np.arange(n-1, -1, -1) if reverse else np.arange(n)
    kdst = pd.DataFrame({'time': time})
    ts, masks = get_time_series_df(n, (kdst.time.min(), kdst.time.max()), kdst)
    assert len(ts) == len(masks) == n, f'{len(ts)}, {len(masks)}, {n}'
    for i, mask in enumerate(masks):
        assert np.count_nonzero(mask) == 1
        index = n-1-i if reverse else i
        assert mask[index] == True
    assert np.all(ts[:-1] < ts[1:])


def test_resolution_definition():
    expected_res  = 3 # % FWHM
    mean   = 123
    sigma  = mean*expected_res/235.48
    values = np.array([np.nan, mean, sigma])
    errors = np.array([np.nan, 0.33*mean, 0.33*sigma])
    expected_ures = expected_res * 0.33 * np.sqrt(2)

    res, ures = resolution(values, errors)

    assert res  == approx(expected_res,  rel=1e-5)
    assert ures == approx(expected_ures, rel=1e-5)

@fixture
def dummy_kr_map():
    return pd.DataFrame(dict(bin = np.arange(10),
                             counts = 1,
                             e0 = 1,
                             ue0 = 0,
                             lt = 1,
                             ult = 0,
                             cov = 0,
                             res_std = 0,
                             pval = 1,
                             in_fid         = True,
                             has_min_counts = True,
                             fit_success = True,
                             valid = True,
                             R = 1,
                             x_index = 1,
                             y_index = 1,
                             Y = 1,
                             X = 1))

@fixture
def dummy_kr_dst():
    n = 10
    res = 5.5 # % FWHM
    lt = 1500
    dv = 1
    X  = np.linspace(-200, 200, n)
    Y  = X
    Z  = np.linspace(400, 500, n)

    X, Y, Z = map(np.ravel, np.meshgrid(X, Y, Z, indexing='ij'))

    DT     = Z/dv
    time   = np.linspace(0, 1, n**3)
    event  = np.arange(n**3)
    mu     = 15000
    sigma  = res*mu/235.48
    energy = np.random.normal(mu, sigma, n**3)
    energy = energy*np.exp(-DT/lt)
    # FIJAR SEED
    pars = dict(res = res, lt = lt, mu = mu, sigma = sigma, dv = dv)

    dst = pd.DataFrame(dict(S1t = 0, S1w = 1, S1h = 2, S1e = 3, S2t = 0, S2w = 4, S2h=5, S2e = energy,
                             S2q = 6, Nsipm = 7, Xrms = 0, Yrms = 0, X = X, Y = Y, DT = DT, time = time,
                             event = event, Z = Z, Zrms = 0, R = 0, Phi = 0, nS1 = 1, nS2 = 1, s1_peak = 0,
                             s2_peak = 0, qmax = 8))
    return dst, pars


def test_computing_kr_parameters(dummy_kr_dst, dummy_kr_map):

    dst, pars_true  = dummy_kr_dst
    pars = computing_kr_parameters(data = dst, ts = 0.5,
                                   emaps = dummy_kr_map,
                                   fittype = KrFitFunction.log_lin,
                                   nbins_dv = 10,
                                   dtrange_dv = [470, 470+10*10],
                                   detector = 'new')

    expected  = dst.mean()
    data_size = len(dst)

    assert pars['ts']    [0] == 0.5
    assert pars['s1w']   [0] == expected.S1w
    assert pars['s1wu']  [0] == 0
    assert pars['s1h']   [0] == expected.S1h
    assert pars['s1hu']  [0] == 0
    assert pars['s1e']   [0] == expected.S1e
    assert pars['s1eu']  [0] == 0
    assert pars['s2w']   [0] == expected.S2w
    assert pars['s2wu']  [0] == 0
    assert pars['s2h']   [0] == expected.S2h
    assert pars['s2hu']  [0] == 0
    assert pars['s2e']   [0] == expected.S2e
    assert pars['s2q']   [0] == expected.S2q
    assert pars['s2qu']  [0] == 0
    assert pars['Nsipm'] [0] == expected.Nsipm
    assert pars['Nsipmu'][0] == 0
    assert pars['Xrms']  [0] == expected.Xrms
    assert pars['Xrmsu'] [0] == 0
    assert pars['Yrms']  [0] == expected.Yrms
    assert pars['Yrmsu'] [0] == 0

    assert np.isclose(pars['s2eu'][0],  dst.S2e.std()/data_size**0.5, rtol=1e-1)
    assert np.isclose(pars['e0'][0],    pars_true['mu'], rtol=5e-2) # E0
    assert np.isclose(pars['lt'][0],    pars_true['lt'], rtol=1e-1) # LT
    assert np.isclose(pars['dv'][0],    pars_true['dv'], rtol=1e-1)
    assert np.isclose(pars['resol'][0], pars_true['res'], rtol=1e-1)
    assert pars['e0u'][0]/pars['e0'][0] <= 0.1
    assert pars['ltu'][0]/pars['lt'][0] <= 0.1
    assert pars['resolu'][0]/pars['resol'][0] <= 0.1
    # assert pars['dvu'][0]/pars['dv'][0] <= 0.1 Uncertainty in dv for this dataset is too high
    # but the validity of the drift velocity computation is tested elsewhere
