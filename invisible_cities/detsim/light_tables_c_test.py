import os
import pytest

import numpy as np

from pytest import mark
from pytest import fixture
from pytest import warns

from .. database.load_db    import DataSiPM
from .. io      .dst_io     import load_dst
from .. core.core_functions import find_nearest

from . light_tables_c import LT_SiPM
from . light_tables_c import LT_PMT
from . light_tables   import create_lighttable_function

from hypothesis                import given
from hypothesis                import settings
from hypothesis.strategies     import floats
from hypothesis.strategies     import integers


@fixture(scope="session")
def default_lt_pmt(get_dfs):
    fname, *_ = get_dfs['lt']
    lt   = LT_PMT(fname=fname)
    npmt = lt.num_sensors
    return lt, npmt


@fixture(scope="session")
def default_lt_sipm(get_dfs):
    fname, *_ = get_dfs['psf']
    datasipm  = DataSiPM("new")
    lt        = LT_SiPM(fname=fname, sipm_database=datasipm)
    nsipm     = lt.num_sensors
    xy        = datasipm[list("XY")].values
    return lt, nsipm, xy


def test_LT_SiPM_optional_arguments(get_dfs):
    datasipm = DataSiPM('new')
    fname, psf_df, psf_conf = get_dfs['psf']
    lt = LT_SiPM(fname=fname, sipm_database=datasipm)

    #check the values are read from the table
    assert lt.el_gap_width  == psf_conf.loc['EL_GAP'    ].astype(float).value
    assert lt.active_radius == psf_conf.loc['ACTIVE_rad'].astype(float).value

    #check optional arguments are set with User Warning
    with warns(UserWarning):
        lt = LT_SiPM(fname=fname, sipm_database=datasipm, el_gap_width=2, active_radius=150)
        assert lt.el_gap_width  == 2
        assert lt.active_radius == 150


@given(xf=floats(min_value=-.25, max_value=.25),
       yf=floats(min_value=-.25, max_value=.25))
def test_LT_SiPM_values_within_active(get_dfs, xf, yf, default_lt_sipm):
    fname, psf_df, psf_conf = get_dfs['psf']
    r_active                = psf_conf.loc['ACTIVE_rad'].astype(float).value
    lt, nsipm, xy_sipm      = default_lt_sipm

    xs  = r_active * xf
    ys  = r_active * yf
    xys = np.array([xs, ys])
    r   = np.sum(xys**2)**0.5

    psfbins = psf_df.index.values
    psf_df  = psf_df / psf_df.shape[1] # some sort of normalization

    indices = np.arange(nsipm)
    dist    = np.sum((xy_sipm - xys)**2, axis=1)**0.5
    for sipm_indx, d in zip(indices, dist):
        if d < psfbins.max():
            psf_bin  = np.digitize(d, psfbins) - 1
            expected = psf_df.loc[psf_bin].values
        else:
            expected = np.zeros(psf_df.shape[1])

        ltvals = lt.get_values(xs, ys, sipm_indx)
        assert np.allclose(ltvals, expected)


def test_LT_SiPM_values_outside_active(get_dfs, default_lt_sipm):
    fname, psf_df, psf_conf = get_dfs['psf']
    r_active = psf_conf.loc['ACTIVE_rad'].astype(float).value

    lt, nsipm, _ = default_lt_sipm
    for sipm_indx in range(nsipm):
        ltvals = lt.get_values(r_active, r_active, sipm_indx)
        assert np.allclose(ltvals, 0)


@given(xf=floats(min_value=-.25, max_value=.25),
       yf=floats(min_value=-.25, max_value=.25))
def test_LT_PMTs_values_within_active(get_dfs, xf, yf, default_lt_pmt):
    fname, lt_df, lt_conf = get_dfs['lt']
    r_active = lt_conf.loc['ACTIVE_rad'].astype(float).value

    xs = r_active * xf
    ys = r_active * yf

    lt, npmt = default_lt_pmt

    lt_xs = np.sort(np.unique(lt_df.index.get_level_values('x')))
    lt_ys = np.sort(np.unique(lt_df.index.get_level_values('y')))
    xs_lt = find_nearest(lt_xs, xs)
    ys_lt = find_nearest(lt_ys, ys)

    expected = lt_df.loc[xs_lt, ys_lt].values[:npmt] # discard sum
    ltvals   = [lt.get_values(xs, ys, pmt)[0] for pmt in range(npmt)]
    assert np.allclose(ltvals, expected)


def test_LT_PMTs_values_outside_active(get_dfs, default_lt_pmt):
    fname, lt_df, lt_conf = get_dfs['lt']
    r_active = lt_conf.loc['ACTIVE_rad'].astype(float).value

    lt, npmt = default_lt_pmt

    expected = np.zeros(npmt)
    ltvals   = [lt.get_values(r_active, r_active, pmt)[0] for pmt in range(npmt)]
    assert np.allclose(ltvals, expected)


@mark.parametrize(   "xf   yf   zero".split(),
                  ( (0.5, 0.5, False), # inside  `r_active_lt`
                    (1.1, 1.1, False), # outside `r_active_lt` but inside  `r_active_forced`
                    (2.0, 2.0,  True), #                           outside `r_active_forced`
                  ))
def test_LT_PMTs_values_extended_r(get_dfs, xf, yf, zero):
    fname, lt_df, lt_conf = get_dfs['lt']
    r_active_lt     = lt_conf.loc['ACTIVE_rad'].astype(float).value
    r_active_forced = 2 * r_active_lt

    with warns(UserWarning):
        lt = LT_PMT(fname=fname,active_radius=r_active_forced)
    npmt = lt.num_sensors

    xs = r_active_lt * xf
    ys = r_active_lt * yf
    if zero:
        expected = np.zeros(npmt)
    else:
        lt_xs = np.sort(np.unique(lt_df.index.get_level_values('x')))
        lt_ys = np.sort(np.unique(lt_df.index.get_level_values('y')))
        xs_lt = find_nearest(lt_xs, xs)
        ys_lt = find_nearest(lt_ys, ys)
        expected = lt_df.loc[xs_lt, ys_lt].values[:npmt] # discard sum

    ltvals = [lt.get_values(xs, ys, pmt)[0] for pmt in range(npmt)]
    assert np.allclose(ltvals, expected)


def test_LTs_non_physical_sensor(get_dfs, default_lt_pmt, default_lt_sipm):
    lt, nsipm, xy_sipm = default_lt_sipm
    values = lt.get_values(0, 0, nsipm) # nsipm sensor ID does not exist
    assert np.all(values==0)

    lt, npmt = default_lt_pmt
    values = lt.get_values(0, 0, npmt) # npmt sensor ID does not exist
    assert np.all(values==0)


@settings(deadline=None)
@given(xs=floats(min_value=-500, max_value=500),
       ys=floats(min_value=-500, max_value=500))
def test_light_tables_pmt_equal_to_function(get_dfs, default_lt_pmt, xs, ys):
    fname, *_ = get_dfs['lt']
    lt, npmt  = default_lt_pmt
    lt_fun    = create_lighttable_function(fname)

    vals_lt  = [lt.get_values(xs,ys, pmtid)[0] for pmtid in range(npmt)]
    vals_fun = lt_fun(np.array([xs]), np.array([ys])).flatten()
    assert np.allclose(vals_fun, vals_lt)
