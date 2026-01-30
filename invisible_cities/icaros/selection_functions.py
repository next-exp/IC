import numpy             as np
import pandas            as pd
from   scipy.optimize     import curve_fit

import matplotlib.pyplot as plt
from invisible_cities.core.core_functions import in_range, shift_to_bin_centers
from scipy import stats
from correction_functions import apply_3Dmap, normalization
import itertools


dtrms2_low = lambda dt: -0.7 + 0.030 * (dt-20) # Gonzalo's
dtrms2_upp = lambda dt: 2.6 + 0.036 * (dt-20) # Gonzalo's2


def eff_of_selection(df_before, df_after, name = ""):
    events_before = df_before.event.nunique()
    events_after = df_after.event.nunique()

    eff = events_after/events_before

    if name:

        print(f"{name} efficiency {eff}, {events_after}/{events_before}.")

    return eff


def select_var_inrange(kdst, col_name, low, high, sel_name):

    sel = in_range(kdst[col_name], *rng)
    df_sel = kdst[sel]

    eff_sel = eff_of_selection(kdst, df_sel, sel_name)

    return df_sel, eff_sel


def select_1S1_1S2(kdst):

    """
    We need to select those events that have 1S1 and 1S2
    after the previous selections, so nS1 and nS2 are no longer valid.

    Group kdst by event, select one variable (time for example)
    and count how many times it is repeated. If it apears only one time
    per event, that event has 1S1 and 1S2.
    """
    #Choosing 'time' column because we had to choose one

    group_kdst = kdst.groupby('event').time.count()

    events_1S1_1S2 = group_kdst[group_kdst == 1].index.values
    df_1S1_1S2 = kdst[kdst.event.isin(events_1S1_1S2)]

    eff_1S1_1S2 = eff_of_selection(kdst, df_1S1_1S2, '1S1 & 1S2')

    return df_1S1_1S2, eff_1S1_1S2



def apply_selections(kdst, dtrms2_low, dtrms2_upp, low_xrays, high_xrays, low_S2t, high_S2t, R_max, low_DT, high_DT, low_nsipm, high_nsipm):

    kdst['Zrms2'] = kdst.Zrms**2

    df, eff_DTband = select_diffusion_band(kdst, 'Zrms2', dtrms2_low(kdst.DT), dtrms2_upp(kdst.DT), 'band selection')

    df, eff_Xrays = select_var_inrange(df, 'Ec', low_xrays, high_xrays, 'remove xrays')

    df, eff_S2t = select_var_inrange(df, 'S2t',  low_S2t, high_S2t, "S2 in trigger time")

    df, eff_1S1_1S2 = select_1S1_1S2(df)

    df, eff_Rmax = select_var_inrage(df, 'R', 0, R_max,  f'events with R less than {R_max}')

    df, eff_DTrange = select_var_inrange(df, 'DT', low_DT, high_DT, f'events in DT range [{low_DT}, {high_DT}]')

    df_final, eff_nsipm = select_var_inrange(df,'Nsipm', low_nsipm, high_nsipm, f'events in NSipm range [{low_nsipm}, {high_nsipm}]')

    total_efficiency = eff_of_selection(kdst, df_final, f'total events after all selections')

    d = {'eff_diffusion_band': eff_DTband, 'eff_Xrays': eff_Xrays, 'eff_1S1_1S2': eff_1S1_1S2,
         'eff_S2_trigger_time': eff_S2t, 'eff_Rmax': eff_Rmax, 'eff_range_DT': eff_DTrange,
         'eff_NSiPMS': eff_nsipm, 'total_efficiency': total_efficiency}

    df_efficiencies = pd.DataFrame(data = d, index = [0])


    return df_final, df_efficiencies
