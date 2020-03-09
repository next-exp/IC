import numpy  as np
import pandas as pd

from typing import Tuple
from typing import List

from ..     core.core_functions import shift_to_bin_centers
from ..     core.core_functions import in_range

def create_psf(pos       : Tuple[np.ndarray, ...],
               charge    : np.ndarray,
               bin_edges : List[np.ndarray],
               ) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray]] :
    """
    Computes the point-spread (PSF) function of a given dataset.

    Parameters
    ----------
    pos       : Hits relative position. Only tested for 2D.
    charge    : Hits SiPM charge normalized to the total peak charge.
    bin_edges : Bin edges for the PSF in each dimension.

    Returns
    ----------
    psf         : Point-spread function.
    entries     : Number of entries per bin in the PSF.:
    bin_centers : Bin centers of the PSF.
    """
    if not len(pos) == len(bin_edges) : raise ValueError         ("Parameter dimensions do not match")
    if len(pos) > 2 :                   raise NotImplementedError(f'{len(pos)}-dimensional PSF not yet implemented')

    entries, _ = np.histogramdd(pos, bin_edges)
    sumC   , _ = np.histogramdd(pos, bin_edges, weights=charge)
    with np.errstate(divide='ignore', invalid='ignore'):
        psf = np.nan_to_num(sumC / entries)

    centers = np.asarray([shift_to_bin_centers(edge) for edge in bin_edges])

    return psf, entries, centers


def add_variable_weighted_mean(df         : pd.DataFrame,
                               var_mean   : str,
                               var_weight : str,
                               mean_name  : str,
                               ) -> pd.DataFrame :
    """
    Adds the average of a variable weighted by another to a
    grouped hits DST 'df' (grouped using groupby, by event id).

    Parameters
    ----------
    df        : dataframe (groupby by event and npeak to do it peak by peak)
    var_mean   : variable to be averaged.
    var_weight : variable to be used as the weight.
    mean_name  : Name of the column containing the average.
    """
    mean, weight = df.loc[:, (var_mean, var_weight)].values.T
    df.loc[:, mean_name] = np.average(mean, weights=weight)


def add_empty_sensors_and_normalize_q(df       : pd.DataFrame,
                                      var      : List[str],
                                      ranges   : List[List[float]],
                                      database : pd.DataFrame
                                      ) -> pd.DataFrame :
    """
    Adds empty sensors to the hDST

    Parameters
    ----------
    df       : dataframe (Containing a single event and npeak)
    var      : dimensions to be considered.
    ranges   : list with the ranges, in each dim, to which empty sensors will be added.
    database : dataframe containing the SiPM database of the detector.

    Returns
    ----------
    df_out   : dst with empty sipm hits.
    """
    delta_x = np.diff(ranges[0])[0]/2
    delta_y = np.diff(ranges[1])[0]/2

    sel_x  = in_range(database.X, df.Xpeak.unique()[0] - delta_x,  df.Xpeak.unique()[0] + delta_x)
    sel_y  = in_range(database.Y, df.Ypeak.unique()[0] - delta_y,  df.Ypeak.unique()[0] + delta_y)

    sensors = database[sel_x & sel_y]

    pd_dict    = {}

    variables  = ['event', 'time', 'npeak']
    variables.extend([f'{v}peak' for v in var])

    pd_dict['X'] = sensors.X
    pd_dict['Y'] = sensors.Y
    pd_dict['Z'] = df.loc[:,'Z'].min()
    for v in variables:
        pd_dict[v] = df.loc[:,v].unique()[0]
    for col in df.columns.values:
        if col not in pd_dict.keys():
            pd_dict[col]  = 0.

    df2 = pd.DataFrame(pd_dict)
    df_out = df.merge(df2, on=list(df), how='outer')
    df_out.drop_duplicates(subset=var, inplace=True, keep='first')
    df_out['NormQ'] = df_out.Q / df_out.Q.sum()
    df_out['nsipm'] = len(df)

    return df_out


def hdst_psf_processing(dsts     : pd.DataFrame,
                        ranges   : List[List[float]],
                        database : pd.DataFrame
                        ) -> pd.DataFrame :
    """
    Adds the necessary info to a hits DST to create the PSF, namely the relative position and the normalized Q.

    Parameters
    ----------
    dsts        : hits (1 SiPM per hit).
    ranges      : range of the PSF in each dimension.
    database    : dataframe containing the SiPM database of the detector.

    Returns
    ----------
    hdst        : hits after processing to create PSF.
    """
    if len(ranges) > 2: raise NotImplementedError(f'{len(ranges)}-dimensional PSF not yet implemented')

    groupedDST    = dsts.groupby(['event', 'npeak'], as_index=False)
    hdst          = groupedDST.apply(add_empty_sensors_and_normalize_q,
                                     ['X', 'Y']     , ranges, database)
    hdst['Zpeak'] = hdst.Z.min()
    hdst['RelX' ] = hdst.X - hdst.Xpeak
    hdst['RelY' ] = hdst.Y - hdst.Ypeak
    hdst['RelZ' ] = 0.

    hdst.reset_index(inplace=True, drop=True)

    return hdst
