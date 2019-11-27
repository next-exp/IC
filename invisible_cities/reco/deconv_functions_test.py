import os
import string
import random
import numpy  as np
import pandas as pd


from hypothesis              import given
from hypothesis              import reproduce_failure
from hypothesis.strategies   import text
from hypothesis.strategies   import floats
from hypothesis.strategies   import tuples
from hypothesis.strategies   import composite
from hypothesis.strategies   import lists
from hypothesis.extra.numpy  import arrays

from .. reco.deconv_functions import cut_and_redistribute_df
from .. reco.deconv_functions import drop_isolated_sensors
from .. reco.deconv_functions import interpolate_signal
from .. reco.deconv_functions import deconvolution_input
from .. reco.deconv_functions import deconvolve
from .. reco.deconv_functions import richardson_lucy
from .. reco.deconv_functions import InterpolationMethod

from .. core.core_functions   import in_range
from .. core.core_functions   import shift_to_bin_centers
from .. core.testing_utils    import assert_dataframes_close

from ..   io.dst_io           import load_dst

from scipy.stats             import multivariate_normal

characters = string.ascii_letters

@composite
def dst(draw, dimension=[3,10]):
    columns = draw(lists(text(characters, min_size=5), min_size=dimension[0], max_size=dimension[1], unique=True))
    values  = draw(lists(arrays(float, 100, floats(1, 1e3, allow_nan=False, allow_infinity=False)), min_size=len(columns), max_size=len(columns)))

    return pd.DataFrame({k:v for k, v in zip(columns, values)})


@given(dst())
def test_cut_and_redistribute_df(df):
    cut_var       = random.choice (df.columns)
    redist_var    = random.choices(df.columns, k=3)
    cut_val       = round(df[cut_var].mean(), 3)
    cut_condition = f'{cut_var} > {cut_val:.3f}'
    cut_function  = cut_and_redistribute_df(cut_condition, redist_var)
    df_cut        = cut_function(df)
    df_cut_manual = df.loc[df[cut_var].values > cut_val, :]
    for var in redist_var:
        df_cut_manual.loc[:, var] = df_cut_manual.loc[:, var] * df.loc[:, var].sum() /  df_cut_manual.loc[:, var].sum()
    assert_dataframes_close(df_cut, df_cut_manual)


def test_drop_isolated_sensors():
    size          = 20
    dist          = [10.1, 10.1]
    x, y          = random.choices(np.linspace(-200, 200, 41), k=size), random.choices(np.linspace(-200, 200, 41), k=size)
    q             = np.random.uniform(0,  20, size)
    e             = np.random.uniform(0, 200, size)
    df            = pd.DataFrame({'X':x, 'Y':y, 'Q':q, 'E':e})
    drop_function = drop_isolated_sensors(dist, ['E'])
    df_cut        = drop_function(df)

    if len(df_cut) > 0:
        assert np.isclose(df_cut.E.sum(), df.E.sum())

    for row in df_cut.itertuples(index=False):
        n_neighbours = len(df_cut[in_range(df_cut.X, row.X - dist[0], row.X + dist[0]) &
                                  in_range(df_cut.Y, row.Y - dist[1], row.Y + dist[1])])
        assert n_neighbours > 1


def test_interpolate_signal():
    ref_interpolation = np.array([0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   ,
                                  0.   , 0.   , 0.   , 0.   , 0.17 , 0.183, 0.188, 0.195, 0.202,
                                  0.2  , 0.2  , 0.19 , 0.181, 0.169, 0.   , 0.   , 0.308, 0.328,
                                  0.344, 0.353, 0.363, 0.365, 0.356, 0.345, 0.327, 0.305, 0.   ,
                                  0.   , 0.501, 0.531, 0.569, 0.583, 0.596, 0.598, 0.583, 0.566,
                                  0.533, 0.496, 0.   , 0.   , 0.693, 0.752, 0.784, 0.819, 0.836,
                                  0.833, 0.82 , 0.794, 0.752, 0.688, 0.   , 0.   , 0.813, 0.882,
                                  0.922, 0.958, 0.976, 0.975, 0.958, 0.927, 0.88 , 0.813, 0.   ,
                                  0.   , 0.812, 0.876, 0.929, 0.958, 0.974, 0.975, 0.959, 0.923,
                                  0.88 , 0.822, 0.   , 0.   , 0.688, 0.752, 0.8  , 0.822, 0.831,
                                  0.833, 0.819, 0.789, 0.753, 0.693, 0.   , 0.   , 0.496, 0.535,
                                  0.567, 0.587, 0.597, 0.591, 0.581, 0.57 , 0.532, 0.504, 0.   ,
                                  0.   , 0.305, 0.326, 0.346, 0.356, 0.362, 0.363, 0.356, 0.342,
                                  0.33 , 0.31 , 0.   , 0.   , 0.168, 0.18 , 0.189, 0.198, 0.202,
                                  0.199, 0.195, 0.192, 0.181, 0.174, 0.   , 0.   , 0.   , 0.   ,
                                  0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   ])

    g = multivariate_normal((0.5, 0.5), (0.05, 0.5))

    grid_x, grid_y = np.mgrid[0:1:12j, 0:1:12j] #Grid for interpolation
    points = np.meshgrid(np.linspace(0, 1, 6), np.linspace(0, 1, 6)) # Coordinates where g is known
    points = (points[0].flatten(), points[1].flatten())
    values = g.pdf(list(zip(points[0], points[1]))) # Value of g at the known coordinates.
    n_interpolation = 12 #How many points to interpolate

    out_interpolation = interpolate_signal(values, points, (np.linspace(-0.05, 1.05, 6), np.linspace(-0.05, 1.05, 6)), [n_interpolation, n_interpolation], InterpolationMethod.cubic)
    inter_charge      = out_interpolation[0].flatten()
    inter_position    = out_interpolation[1]
    ref_position = shift_to_bin_centers(np.linspace(-0.05, 1.05, n_interpolation + 1))

    assert np.allclose(ref_interpolation, np.around(inter_charge, decimals=3))
    assert np.allclose(ref_position     , sorted(set(inter_position[0])))
    assert np.allclose(ref_position     , sorted(set(inter_position[1])))


def test_deconvolution_input(data_hdst, data_hdst_deconvolved):
    ref_interpolation = np.load(data_hdst_deconvolved)
    hdst              = load_dst(data_hdst, 'RECO', 'Events')

    h = hdst[(hdst.event == 3021916) & (hdst.npeak == 0)]
    h = h.groupby(['X', 'Y']).Q.sum().reset_index()
    h = h[h.Q>40]

    interpolator = deconvolution_input([10., 10.], [1., 1.], InterpolationMethod.cubic)
    inter        = interpolator((h.X, h.Y), h.Q)

    assert np.allclose(ref_interpolation['e_inter'], inter[0])
    assert np.allclose(ref_interpolation['x_inter'], inter[1][0])
    assert np.allclose(ref_interpolation['y_inter'], inter[1][1])


def test_deconvolve(data_hdst, data_hdst_deconvolved):
    ref_interpolation = np.load (data_hdst_deconvolved)
    hdst              = load_dst(data_hdst, 'RECO', 'Events')

    h = hdst[(hdst.event == 3021916) & (hdst.npeak == 0)]
    z = h.Z.mean()
    h = h.groupby(['X', 'Y']).Q.sum().reset_index()
    h = h[h.Q>40]

    deconvolutor = deconvolve(15, 0.01, [10., 10.], [1., 1.], inter_method=InterpolationMethod.cubic)

    x, y   = np.linspace(-49.5, 49.5, 100), np.linspace(-49.5, 49.5, 100)
    xx, yy = np.meshgrid(x, y)
    xx, yy = xx.flatten(), yy.flatten()

    psf           = {}
    psf['factor'] = multivariate_normal([0., 0.], [1.027 * np.sqrt(z/10)] * 2).pdf(list(zip(xx, yy)))
    psf['xr']     = xx
    psf['yr']     = yy
    psf['zr']     = [z] * len(xx)
    psf           = pd.DataFrame(psf)

    deco          = deconvolutor((h.X, h.Y), h.Q, psf)

    assert np.allclose(ref_interpolation['e_deco'], deco[0].flatten())
    assert np.allclose(ref_interpolation['x_deco'], deco[1][0])
    assert np.allclose(ref_interpolation['y_deco'], deco[1][1])


def test_richardson_lucy(data_hdst, data_hdst_deconvolved):
    ref_interpolation = np.load (data_hdst_deconvolved)
    hdst              = load_dst(data_hdst, 'RECO', 'Events')
    h = hdst[(hdst.event == 3021916) & (hdst.npeak == 0)]
    z = h.Z.mean()
    h = h.groupby(['X', 'Y']).Q.sum().reset_index()
    h = h[h.Q>40]
    interpolator = deconvolution_input([10., 10.], [1., 1.], InterpolationMethod.cubic)
    inter = interpolator((h.X, h.Y), h.Q)

    x, y = np.linspace(-49.5, 49.5, 100), np.linspace(-49.5, 49.5, 100)
    xx, yy = np.meshgrid(x, y)
    xx, yy = xx.flatten(), yy.flatten()
    psf = {}
    psf['factor'] = multivariate_normal([0., 0.], [1.027 * np.sqrt(z/10)] * 2).pdf(list(zip(xx, yy)))
    psf['xr']     = xx
    psf['yr']     = yy
    psf['zr']     = [z] * len(xx)
    psf   = pd.DataFrame(psf)

    deco = richardson_lucy(inter[0], psf.factor.values.reshape(psf.xr.nunique(), psf.yr.nunique()).T, iterations=15, iter_thr=0.0001)

    assert np.allclose(ref_interpolation['e_deco'], deco.flatten())
