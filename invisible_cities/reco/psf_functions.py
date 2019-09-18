import numpy  as np
import pandas as pd

from typing import Tuple
from typing import List

from ..core.core_functions import shift_to_bin_centers

def createPSF(pos    : Tuple[np.ndarray, ...],
              charge : np.ndarray,
              nbins  : int,
              ranges : List[List[float]]
              ) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray]] :
    """
    Computes the point-spread (PSF) function of a given dataset.

    Parameters
    ----------
    pos    : Hits relative position. Only tested for 2D.
    charge : Hits SiPM charge normalized to the total peak charge.
    nbins  : The number of bins in each dimension.
    ranges : Range of the PSF in each dimension.

    Returns
    ----------
    psf         : Point-spread function.
    entries     : Number of entries per bin in the PSF.:
    bin_centers : Bin centers of the PSF.
    """

    entries, edges = np.histogramdd(pos, nbins, range=ranges, normed=False)
    sumC   , edges = np.histogramdd(pos, nbins, range=ranges, normed=False, weights=charge)
    with np.errstate(divide='ignore', invalid='ignore'):
        psf = np.nan_to_num(sumC/entries)

    return psf, entries, [shift_to_bin_centers(edge) for edge in edges]


def hdst_PSF_processing(dsts        : pd.DataFrame,
                        ranges      : List[List[float]],
                        sampleWidth : List[float]
                        ) -> pd.DataFrame :
    """
    Adds the necessary info to a hits DST to create the PSF, namely the relative position and the normalized Q.

    Parameters
    ----------
    dsts        : hits (1 SiPM per hit).
    ranges      : range of the PSF in each dimension.
    sampleWidth : Sampling distance of the hits.

    Returns
    ----------
    hdst        : hits after processing to create PSF.
    """
    def AddVariableWeightedMean(df        : pd.DataFrame,
                                varMean   : str,
                                varWeight : str
                                ) -> pd.DataFrame :
        """
        Adds the average of a variable weighted by another to a
        grouped hits DST 'df' (grouped using groupby, by event id).

        Parameters
        ----------
        df        : groupby by event and npeak.
        varMean   : variable to be averaged.
        varWeight : variable to be uses as the weight.

        Returns
        ----------
        df        : dst with the weighted average.
        """
        df[varMean + 'peak'] =  (df[varMean]*df[varWeight]).sum()/df[varWeight].sum()
        return(df)

    def AddEmptySensors(df  : pd.DataFrame,
                        var : List[str]
                        ) -> pd.DataFrame :
        """
        Adds empty sensors to the hDST

        Parameters
        ----------
        df  : groupby by event and npeak.
        var : dimensions to be considered.

        Returns
        ----------
        df  : dst with empty sipm hits.
        """
        distance = (np.diff(ranges)/2).flatten()
        means    = [int(df[f'{v}peak'].mean()) for v in var[:len(ranges)]]
        means    = [mean - mean%sampleWidth[i]  for i, mean in enumerate(   means)]
        varrange = [[means[i] - d, means[i] + d + sampleWidth[i]]                   for i,    d in enumerate(distance)]
        allbins  = [int(np.diff(rang)/sampleWidth[i])              for i, rang in enumerate(varrange)]
        Hs, edges   = np.histogramdd(tuple(df[v].values for v in var[:len(ranges)]), bins=allbins, range=varrange, normed=False, weights=df['Q'])
        interPoints = np.meshgrid(*(shift_to_bin_centers(edge) for edge in edges), indexing='ij')
        if len(ranges) < 3:
            interPoints.append(np.array([df['Z'].min()] * len(interPoints[0].flatten())))

        pd_dict1 = {f'{v     }' : interPoints[i].flatten() for i, v in enumerate(var)}
        pd_dict2 = {f'{var[i]}peak' : [df[f'{var[i]}peak'].mean()] * len(interP.flatten())
                    for i, interP in enumerate(interPoints[:len(ranges)])}

        pd_dict3 = {}
        pd_dict3['Q'] = Hs.flatten()
        pd_dict3['NormQ'] = Hs.flatten() / Hs.sum()
        pd_dict3['E'] = pd_dict3['NormQ'] * df['E'].sum()
        pd_dict3['nsipm'] = [len(df)] * len(Hs.flatten())
        pd_dict4 = {k : [df[k].min()] * len(interPoints[0].flatten()) for k in ['event', 'time', 'npeak']}
        pd_dict  = {**pd_dict1, **pd_dict2, **pd_dict3, **pd_dict4}

        return pd.DataFrame(pd_dict).sort_values(['event', 'npeak', 'E'], ascending=[1, 1, 0]).reindex(list(df.columns).append('NormQ'), axis=1)

    groupedDST = dsts.groupby(['event', 'npeak'], as_index=False)
    if len(ranges) >= 3:
        hdst          = groupedDST.apply(AddVariableWeightedMean, 'Z', 'E')
        hdst          = hdst.groupby(['event', 'npeak'], as_index=False).apply(AddEmptySensors, ['X', 'Y', 'Z']).reset_index(drop=True)
        hdst['RelZ']  = hdst.Z - hdst.Zpeak
    else:
        hdst          = groupedDST.apply(AddEmptySensors, ['X', 'Y', 'Z']).reset_index(drop=True)
        #hdst          = dsts
        #print(hdst)
        hdst['Zpeak'] = [hdst.Z.min()] * len(hdst)
        hdst['RelZ']  = [0           ] * len(hdst)

    hdst._is_copy = False

    hdst['RelX' ]      = hdst.X - hdst.Xpeak
    hdst['RelY' ]      = hdst.Y - hdst.Ypeak

    return hdst
