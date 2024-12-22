import numpy                                as np
import numpy.testing                        as npt

from   pytest                               import approx
from   flaky                                import flaky

from   hypothesis                           import given
from   hypothesis.strategies                import floats
from .. reco.krmap_evolution                import sigmoid, compute_drift_v, resolution

from .. database                            import load_db  as DB

sensible_floats = floats(-1e4, +1e4)


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


@flaky(max_runs=10, min_passes=9)
def test_compute_drift_v_when_moving_edge():
    edge    = np.random.uniform(530, 570)
    Nevents = 100 * 1000
    data    = np.random.uniform(450, edge, Nevents)
    data    = np.random.normal(data, 1)
    dv, dvu = compute_drift_v(data, 60, [500,600],
                              [1500, 550,1,0], 'next100')
    dv_th   = DB.DetectorGeo('next100').ZMAX[0]/edge

    assert dv_th == approx(dv, abs=5*dvu)


def test_compute_drift_v_failing_fit_return_nan():
    dst     = np.random.rand(1000)*1200
    dv_vect = compute_drift_v(dst, 60, [500,600], [1500, 550, 1, 0], 'new')
    nans    = np.array([np.nan, np.nan])
    assert dv_vect == nans


def test_resolution_typical_case():

    values = np.array([10, 100, 2])
    errors = np.array([0.1, 1, 0.05])
    expected_res  = 235.48 * (values[2] / values[1])  # 235.48 * sigma/mu
    expected_ures = expected_res * ((errors[1] / values[1])**2 + (errors[2] / values[2])**2)**0.5

    res, ures = resolution(values, errors)

    assert res  == approx(expected_res,  rel=1e-5)
    assert ures == approx(expected_ures, rel=1e-5)
