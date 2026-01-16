import numpy             as np
import pandas            as pd
from   scipy.optimize     import curve_fit

import matplotlib.pyplot as plt
from invisible_cities.core.core_functions import in_range, shift_to_bin_centers
from scipy import stats
from correction_functions import apply_3Dmap, normalization
from pathlib import Path
import itertools


dtrms2_low = lambda dt: -0.7 + 0.030 * (dt-20) # Gonzalo's
dtrms2_upp = lambda dt: 2.6 + 0.036 * (dt-20) # Gonzalo's2
dtrms2_cen = lambda dt:  1.0 + 0.033 * (dt-20)
def dist_to_bandcenter(df): return df.Zrms**2 - dtrms2_cen(df.DT)


def load_files(path):
    return [str(p) for p in Path(path).rglob("*.h5")]


def eff_of_selection(df_before, df_after, name = ""):
    events_before = df_before.event.nunique()
    events_after = df_after.event.nunique()

    eff = events_after/events_before

    if name:

        print(f"{name} efficiency {eff}, {events_after}/{events_before}.")

    return eff


def apply_correctionmap(kdst, map3D, norm_method, xy_params, keV = True):
    
    corrected_energy = apply_3Dmap(map3D, norm_method, kdst.DT, kdst.X, kdst.Y, kdst.S2e, xy_params = xy_params, keV = keV)
    
    col_name = 'Ec' if 'Ec' not in kdst.columns else 'Ec_2'
    kdst[col_name] = corrected_energy.values

    return kdst


def select_diffusion_band(kdst, dtrms2_low, dtrms2_upp):

    sel_DTband = in_range(kdst.Zrms**2, dtrms2_low(kdst.DT), dtrms2_upp(kdst.DT))
    df_DTband = kdst[sel_DTband]

    eff_DT = eff_of_selection(kdst, df_DTband, 'band selection')

    return df_DTband, eff_DT


def select_Xrays(kdst, low_xrays, high_xrays):

    sel_xrays = in_range(kdst.Ec, low_xrays, high_xrays)
    df_Xrays = kdst[sel_xrays]

    eff_Xrays = eff_of_selection(kdst, df_Xrays, 'remove xrays')

    return df_Xrays, eff_Xrays


def select_1S1_1S2(kdst):

    """
    We need to select those events that have 1S1 and 1S2
    after the previous selections, so nS1 and nS2 are no longer valid.

    Group kdst by event, select one variable (time for example)
    and count how many times it is repeated. If it apears only one time
    per event, that event has 1S1 and 1S2.
    """

    group_kdst = kdst.groupby('event')['time'].count()

    events_1S1_1S2 = group_kdst[group_kdst == 1].index.values
    df_1S1_1S2 = kdst[kdst.event.isin(events_1S1_1S2)]

    eff_1S1_1S2 = eff_of_selection(kdst, df_1S1_1S2, '1S1 & 1S2')

    return df_1S1_1S2, eff_1S1_1S2


def select_S2t(kdst, low_S2t, high_S2t):

    sel_S2t = in_range(kdst.S2t, low_S2t, high_S2t)
    df_S2t = kdst[sel_S2t]

    eff_S2t = eff_of_selection(kdst, df_S2t, "S2 in trigger time")

    return df_S2t, eff_S2t


def select_Rmax(kdst, R_max):
    df_Rmax = kdst[kdst.R <= R_max]

    eff_Rmax = eff_of_selection(kdst, df_Rmax, f'events with R less than {R_max}')

    return df_Rmax, eff_Rmax


def select_DTrange(kdst, low_DT, high_DT):
    df_DTrange = kdst[(kdst.DT >= low_DT) & (kdst.DT <= high_DT)]

    eff_DTrange = eff_of_selection(kdst, df_DTrange, f'events in DT range [{low_DT}, {high_DT}]')

    return df_DTrange, eff_DTrange


def select_nsipm(kdst, low_nsipm, high_nsipm):
    sel_nsipm = (kdst.Nsipm >= low_nsipm) & (kdst.Nsipm <= high_nsipm)
    df_nsipm = kdst[sel_nsipm]

    eff_nsipm = eff_of_selection(kdst, df_nsipm, f'events in NSipm range [{low_nsipm}, {high_nsipm}]')

    return df_nsipm, eff_nsipm


def apply_selections(kdst, run_number,  dtrms2_low, dtrms2_upp, low_xrays, high_xrays, low_S2t, high_S2t, R_max, low_DT, high_DT, low_nsipm, high_nsipm):

    df, eff_DTband = select_diffusion_band(kdst, dtrms2_low, dtrms2_upp)

    df, eff_Xrays = select_Xrays(df, low_xrays, high_xrays)

    df, eff_1S1_1S2 = select_1S1_1S2(df)

    df, eff_S2t = select_S2t(df, low_S2t, high_S2t)

    df, eff_Rmax = select_Rmax(df, R_max)

    df, eff_DTrange = select_Dtrange(df, low_DT, high_DT)

    df_final, eff_nsipm = select_nsipm(df, low_nsipm, high_nsipm)

    total_efficiency = eff_of_selection(kdst, df_final, f'total events after all selections')

    d = {'eff diffusion band': [eff_DT], 'eff X rays': [eff_Xrays], 'eff 1S1 & 1S2': [eff_1S1_1S2],
         'eff S2 trigger time': [eff_S2t], 'eff Rmax': [eff_Rmax], 'eff range DT': [eff_DTrange],
         'eff number of SiPMS': [eff_nsipm], 'total efficiency': [total_eff]}

    df_efficiencies = pd.DataFrame(data = d)

    return df_final, df_efficiencies
