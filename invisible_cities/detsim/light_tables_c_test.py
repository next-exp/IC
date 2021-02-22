import os
import pytest

import numpy as np

from pytest import warns

from .. database.load_db    import DataSiPM
from .. io      .dst_io     import load_dst
from .. core.core_functions import find_nearest

from . light_tables_c import LT_SiPM
from . light_tables_c import LT_PMT

from hypothesis                import given
from hypothesis                import settings
from hypothesis.strategies     import floats
from hypothesis.strategies     import integers


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

@given(xs=floats(min_value=-500, max_value=500),
       ys=floats(min_value=-500, max_value=500),
       sipm_indx=integers(min_value=0, max_value=1500))
def test_LT_SiPM_values(get_dfs, xs, ys, sipm_indx):
    datasipm = DataSiPM('new')
    fname, psf_df, psf_conf = get_dfs['psf']

    r_active = psf_conf.loc['ACTIVE_rad'].astype(float).value
    r = np.sqrt(xs**2 + ys**2)
    psfbins = psf_df.index.values
    lenz = psf_df.shape[1]
    psf_df = psf_df /lenz
    lt = LT_SiPM(fname=fname, sipm_database=datasipm)
    x_sipm, y_sipm = datasipm.iloc[sipm_indx][['X', 'Y']]
    dist = np.sqrt((xs-x_sipm)**2+(ys-y_sipm)**2)
    psf_bin = np.digitize(dist, psfbins)-1
    max_psf = psf_df.index.max()
    if (dist>=max_psf) or (r>=r_active):
        values = np.zeros(psf_df.shape[1])
    else:
        values = (psf_df.loc[psf_bin].values)

    ltvals = lt.get_values(xs, ys, sipm_indx)
    np.testing.assert_allclose(values, ltvals)



@given(xs=floats(min_value=-500, max_value=500),
       ys=floats(min_value=-500, max_value=500),
       pmt_indx=integers(min_value=0, max_value=11))
def test_LT_PMTs_values(get_dfs, xs, ys, pmt_indx):
    fname, lt_df, lt_conf = get_dfs['lt']
    r_active = lt_conf.loc['ACTIVE_rad'].astype(float).value
    r = np.sqrt(xs**2 + ys**2)

    lt = LT_PMT(fname=fname)

    xs_lt =  find_nearest(np.sort(np.unique(lt_df.index.get_level_values('x'))), xs)
    ys_lt =  find_nearest(np.sort(np.unique(lt_df.index.get_level_values('y'))), ys)
    if (r>=r_active):
        values = np.array([0]) #the values are one dimension only
    else:
        values = lt_df.loc[xs_lt, ys_lt].values[pmt_indx]
    ltvals = lt.get_values(xs, ys, pmt_indx)
    np.testing.assert_allclose(values, ltvals)


@given(xs=floats(min_value=-500, max_value=500),
       ys=floats(min_value=-500, max_value=500),
       pmt_indx=integers(min_value=0, max_value=11))
def test_LT_PMTs_values_extended_r(get_dfs, xs, ys, pmt_indx):
    fname, lt_df, lt_conf = get_dfs['lt']
    r_active = lt_conf.loc['ACTIVE_rad'].astype(float).value
    r_new = 2*r_active
    r = np.sqrt(xs**2 + ys**2)
    with warns(UserWarning):
        lt = LT_PMT(fname=fname,active_radius=r_new)
    xs_lt =  find_nearest(np.sort(np.unique(lt_df.index.get_level_values('x'))), xs)
    ys_lt =  find_nearest(np.sort(np.unique(lt_df.index.get_level_values('y'))), ys)
    if (r>=r_new):
        values = np.array([0]) #the values are one dimension only
    else:
        values = lt_df.loc[xs_lt, ys_lt].values[pmt_indx]
    ltvals = lt.get_values(xs, ys, pmt_indx)
    np.testing.assert_allclose(values, ltvals)


def test_LTs_non_physical_sensor(get_dfs):
    datasipm = DataSiPM('new')
    fname, psf_df, psf_conf = get_dfs['psf']
    lt = LT_SiPM(fname=fname, sipm_database=datasipm)
    xs = 0
    ys = 0
    sipm_id = len(datasipm)
    values = lt.get_values(xs, ys, sipm_id)
    assert all(values==0)

    fname, lt_df, lt_conf = get_dfs['lt']
    lt = LT_PMT(fname=fname)
    pmt_id = 12
    values = lt.get_values(xs, ys,  pmt_id)
    assert all(values==0)


from .light_tables import create_lighttable_function
@given(xs=floats(min_value=-500, max_value=500),
       ys=floats(min_value=-500, max_value=500))
def test_light_tables_pmt_equal(get_dfs, xs, ys):
    fname, lt_df, lt_conf = get_dfs['lt']
    lt_c  = LT_PMT(fname=fname)
    s2_lt = create_lighttable_function(fname)
    pmts_ids  = np.arange(lt_c.num_sensors)
    vals_lt_c = np.concatenate([lt_c.get_values(xs,ys, pmtid)for pmtid in pmts_ids]).flatten()
    vals_lt   = s2_lt(np.array([xs]), np.array([ys])).flatten()
    np.testing.assert_allclose(vals_lt_c, vals_lt)
