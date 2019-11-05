import warnings
import numpy  as np
import pandas as pd
from typing import Union
from typing import List
from .  corrections import Correction
from .  corrections import LifetimeXYCorrection
from .. io.dst_io   import load_dst
from .. io.dst_io   import load_dsts


def load_z_corrections(filename):
    dst = load_dst(filename, "Corrections", "Zcorrections")
    return Correction((dst.z.values,), dst.factor.values, dst.uncertainty.values)


def load_xy_corrections(filename, *,
                        group = "Corrections",
                        node  = "XYcorrections",
                        **kwargs):
    dst  = load_dst(filename, group, node)
    x, y = np.unique(dst.x.values), np.unique(dst.y.values)
    f, u = dst.factor.values, dst.uncertainty.values

    return Correction((x, y),
                      f.reshape(x.size, y.size),
                      u.reshape(x.size, y.size),
                      **kwargs)


def load_lifetime_xy_corrections(filename, *,
                                 group = "Corrections",
                                 node  = "LifetimeXY",
                                 scale = 1,
                                 **kwargs):
    """
    Load the lifetime map from hdf5 file.

    Parameters
    ----------
    filename: str
        Path to the file containing the map.
    group: str
        Name of the group where the table is stored.
    node: str
        Name of the table containing the data.
    scale: float
        Scale factor for the lifetime values.

    Other kwargs are passed to the contructor of LifetimeXYCorrection.
    """
    dst  = load_dst(filename, group, node)
    x, y = np.unique(dst.x.values), np.unique(dst.y.values)
    f, u = dst.factor.values * scale, dst.uncertainty.values * scale

    return LifetimeXYCorrection(f.reshape(x.size, y.size),
                                u.reshape(x.size, y.size),
                                x, y, **kwargs)

def dst_event_id_selection(data, event_ids):
    """Filter a DST by a list of event IDs.
    Parameters
    ----------
    data      : pandas DataFrame
        DST to be filtered.
    event_ids : list
        List of event ids that will be kept.
    Returns
    -------
    filterdata : pandas DataFrame
        Filtered DST
    """
    if 'event' in data:
        sel = np.isin(data.event.values, event_ids)
        return data[sel]
    else:
        print(r'DST does not have an "event" field. Data returned is unfiltered.')
        return data

def load_paolina_summary_(filename : str) -> pd.DataFrame:
    """ Merges information from DST/Events, RUN/events and PAOLINA/Summary table into one DataFrame.
    Parameters
    ----------
    filename      : str
        .h5 ouput of Esmeralda
    Returns
    -------
    merged_DF : pandas DataFrame
        Extended summary information
    """
    summary    = load_dst(filename, 'PAOLINA', 'Summary')
    kdst       = load_dst(filename, 'DST'    , 'Events' )
    event_info = load_dst(filename, 'Run'    , 'events' )

    def get_s1_info(df_series):
        if(len(df_series.unique()) != 1):
            warnings.warn("Number of recorded S1 energies differs from 1 per event.Choosing first S1", UserWarning)
        return df_series.iloc[0]
    #per event we extract S1 energy, time and nS2 information, sum of S2 energy and charge
    kdst_to_merge = kdst.loc[:,['event', 'S1e', 'S1t', 'nS2', 'S2e', 'S2q']].groupby('event').agg({'S1e' : get_s1_info,
                                                                                                   'S1t' : get_s1_info,
                                                                                                   'nS2' : get_s1_info,
                                                                                                   'S2e' : np.sum,
                                                                                                   'S2q' : np.sum}).reset_index()
    #have to rename columns to match old event summary style
    kdst_to_merge.rename(columns={"S2e": "S2e0", "S2q":  "S2q0"}, inplace=True)
    event_info   .rename(columns={"evt_number": "event", "timestamp": "time"}, inplace=True)
    #merge event_info, kdst info and summary into one dataframe
    extended_summary  = summary.merge(kdst_to_merge,
                                      on='event', how='left').merge(event_info,
                                                                    on='event', how='left')
    return extended_summary


def load_paolina_summary(filenames : Union[str, List[str]]) -> pd.DataFrame :
    """ Merges information from DST/Events, RUN/events and PAOLINA/Summary table into one DataFrame.
    Parameters
    ----------
    filename      : str or List[str]
        .h5 ouput of Esmeralda
    Returns
    -------
    merged_DF : pandas DataFrame
        Extended summary information
    """
    if not isinstance(filenames, list):
        return load_paolina_summary_(filenames)
    else:
        return pd.concat([load_paolina_summary_(filename) for filename in filenames], ignore_index=True)
