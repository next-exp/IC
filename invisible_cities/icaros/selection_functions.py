import numpy             as np
import pandas            as pd
from   scipy.optimize     import curve_fit
from typing import Callable, Union

from .. core           .core_functions import in_range, shift_to_bin_centers
from scipy import stats
from .. icaros         .correction_functions import apply_3Dmap, normalization
import itertools


def eff_of_selection(df_before  : pd.DataFrame,
                     df_after   : pd.DataFrame,
                     name       : str = "") -> float:
    """
    Calculates the efficiency of a certain selection.
    Parameters
    ----------
    df_before : pd.DataFrame
        Dataframe (kdst sophronia output) before applying selections
    df_after : pd.DataFrame
        Dataframe after applying selections
    name : string (optional)
        Name of the selection
    Returns
    -------
    eff : float
        Efficiency of the selection
    """

    events_before = df_before.event.nunique()
    events_after = df_after.event.nunique()

    eff = events_after/events_before

    if name:

        print(f"{name} efficiency {eff}, {events_after}/{events_before}.")

    return eff


def select_var_inrange(kdst      : pd.DataFrame,
                       col_name  : str,
                       low       : float,
                       high      : float,
                       sel_name  : str) -> [pd.DataFrame, float]:
    """
    Selects a certain variable from a dataframe in a certain range.
    Parameters
    ----------
    kdst : pd.DataFrame
       Dataframe (kdst sophronia output) where the selection is being applied
    col_name : str
       Name of the variable that is being selected
    low : float
       Low limit of the range inside the variable is being selected
    high : float
       Upper limit fo the range inside the variable is being selected
    sel_name : str
       Selection name for efficiency calculation
    Returns
    -------
    df_sel : pd.DataFrame
       Dataframe after applying selection
    eff_sel : str
       Efficiency of the performed selection
    """

    sel = in_range(kdst[col_name], low, high)
    df_sel = kdst[sel]

    eff_sel = eff_of_selection(kdst, df_sel, sel_name)

    return df_sel, eff_sel


def select_1S1_1S2(kdst : pd.DataFrame) -> [pd.DataFrame, float]:

    """
    We need to select those events that have 1S1 and 1S2
    after the previous selections, so nS1 and nS2 are no longer valid.

    Group kdst by event, select one variable (time for example)
    and count how many times it is repeated. If it apears only one time
    per event, that event has 1S1 and 1S2.

    Parameters
    ----------
    kdst : pd.DataFrame
      Dataframe (kdst sophronia output) where the 1S1&1S2 selection is being applied
    Returns
    -------
    df_1S1_1S2 : pd.DataFrame
      Dataframe after applying 1S1&1S2 selection
    eff_1S1_1S2 : str
      Efficiency of the 1S1&1S2 selection
    """

    #Choosing 'time' column because we had to choose one
    group_kdst = kdst.groupby('event').time.count()

    events_1S1_1S2 = group_kdst[group_kdst == 1].index.values
    df_1S1_1S2 = kdst[kdst.event.isin(events_1S1_1S2)]

    eff_1S1_1S2 = eff_of_selection(kdst, df_1S1_1S2, '1S1 & 1S2')

    return df_1S1_1S2, eff_1S1_1S2



def apply_selections(kdst        : pd.DataFrame,
                     dtrms2_low  : Callable,
                     dtrms2_upp  : Callable,
                     low_xrays   : float,
                     high_xrays  : float,
                     low_S2t     : float,
                     high_S2t    : float,
                     R_max       : float,
                     low_DT      : float,
                     high_DT     : float,
                     low_nsipm   : float,
                     high_nsipm  : float) -> [pd.DataFrame, pd.DataFrame]:
    """
    Applies all necessary selections to get a selected dataframe with its corresponding efficiencies.
    Parameters
    ----------
    kdst : pd.DataFrame
      Dataframe (kdst sophronia output) where the selections are being applied
    dtrms2_low : function
      Low limit for drift time (DT) when selecting diffusion band
    dtrms2_upp : function
      Upper limit for drift time (DT) when selecting diffusion band
    low_xrays : float
      Low limits for corrected energy (keV) to avoid Kr Xrays
    high_xrays : float
      Upper limit for corrected energy (keV) to avoid Kr Xrays
    low_S2t : float
      Low limit for S2 trigger time
    high_S2t : float
      Upper limit for S2 trigger time
    R_max : float
      Maximum radius
    low_DT : float
      Low drift time limit
    high_DT : float
      Upper drift time limit
    low_nsipm : float
g      Low number of triggered SiPMs
    high_nsipm : floar
      Upper number of triggered SiPMs
    Returns
    -------
    df_final : pd.DataFrame
      Dataframe after all selections
    df_efficiencies : pd.DataFrame
      Dataframe containing the efficiency of each selection and the total efficiency
    """

    kdst['Zrms2'] = kdst.Zrms**2

    df, eff_DTband = select_var_inrange(kdst, 'Zrms2', dtrms2_low(kdst.DT), dtrms2_upp(kdst.DT), 'band selection')

    df, eff_Xrays = select_var_inrange(df, 'Ec', low_xrays, high_xrays, 'remove xrays')

    df, eff_S2t = select_var_inrange(df, 'S2t',  low_S2t, high_S2t, "S2 in trigger time")

    df, eff_1S1_1S2 = select_1S1_1S2(df)

    df, eff_Rmax = select_var_inrange(df, 'R', 0, R_max,  f'events with R less than {R_max}')

    df, eff_DTrange = select_var_inrange(df, 'DT', low_DT, high_DT, f'events in DT range [{low_DT}, {high_DT}]')

    df_final, eff_nsipm = select_var_inrange(df,'Nsipm', low_nsipm, high_nsipm, f'events in NSipm range [{low_nsipm}, {high_nsipm}]')

    total_efficiency = eff_of_selection(kdst, df_final, f'total events after all selections')

    d = {'eff_diffusion_band': eff_DTband, 'eff_Xrays': eff_Xrays, 'eff_1S1_1S2': eff_1S1_1S2,
         'eff_S2_trigger_time': eff_S2t, 'eff_Rmax': eff_Rmax, 'eff_range_DT': eff_DTrange,
         'eff_NSiPMS': eff_nsipm, 'total_efficiency': total_efficiency}

    df_efficiencies = pd.DataFrame(data = d, index = [0])


    return df_final, df_efficiencies
