import numpy  as np
import pandas as pd

from itertools   import compress
from copy        import deepcopy
from typing      import List

from .. evm  import event_model as evm
from .. types.ic_types      import NN
from .. types.ic_types      import xy

EPSILON = np.finfo(np.float64).eps


def merge_NN_hits(hits : pd.DataFrame, same_peak : bool = True) -> pd.DataFrame:
    """ Returns a list of the hits where the  energies of NN hits are distributed to the closest hits such that the added energy is proportional to
    the hit energy. If all the hits were NN the function returns empty list. """
    sel = hits.Q == NN
    if not np.any(sel): return hits

    nn_hits = hits.loc[ sel]
    hits    = hits.loc[~sel].copy()

    corrections = pd.DataFrame(dict(E=0, Ec=0), index=hits.index.values)
    for _, nn_hit in nn_hits.iterrows():
        candidates = hits.loc[hits.npeak == nn_hit.npeak] if same_peak else hits
        if len(candidates) == 0: continue # drop hit !!! dangerous

        dz      = np.abs(candidates.Z - nn_hit.Z)
        closest = candidates.loc[np.isclose(dz, dz.min())]
        index   = closest.index

        corr_e  = nn_hit.E  * closest.E  / closest.E .sum()
        corr_ec = nn_hit.Ec * closest.Ec / closest.Ec.sum()
        corrections.loc[index, "E Ec".split()] += np.stack([corr_e, corr_ec], axis=1)

    # correction factors based on original charge values, this is why
    # we accumulate corrections, which is order insensitive
    hits.loc[:, "E Ec".split()] += corrections.values
    return hits


def empty_hit(event, timestamp, peak_no, x_peak, y_peak, z, e, ec):
    return pd.DataFrame(dict( event    = event
                            , time     = timestamp
                            , npeak    = peak_no
                            , Xpeak    = x_peak
                            , Ypeak    = y_peak
                            , nsipm    = 1
                            , X        = NN
                            , Y        = NN
                            , Xrms     = 0
                            , Yrms     = 0
                            , Z        = z
                            , Q        = NN
                            , E        = e
                            , Qc       = -1
                            , Ec       = ec
                            , track_id = -1
                            , Ep       = -1), index=[0])


def apply_threshold(hits : pd.DataFrame, th : float, on_corrected : bool=False) -> pd.DataFrame:
    raw_e_slice = hits.E.sum()
    cor_e_slice = np.nansum(hits.Ec) + np.finfo(np.float64).eps

    col         = "Qc" if on_corrected else "Q"
    mask_thresh = hits[col] >= th

    if not mask_thresh.any():
        first = hits.iloc[0]
        return empty_hit( first.event, first.time
                        , first.npeak, first.Xpeak, first.Ypeak
                        , first.Z, raw_e_slice, cor_e_slice)

    hits = hits.loc[mask_thresh].copy()
    qsum = np.nansum(hits.Q) + EPSILON
    hits.loc[:, "E" ] = hits.Q / qsum * raw_e_slice
    hits.loc[:, "Ec"] = hits.Q / qsum * cor_e_slice
    return hits


def threshold_hits(hits : pd.DataFrame, th : float, on_corrected : bool=False) -> pd.DataFrame:
    """Returns list of the hits which charge is above the threshold. The energy of the hits below the threshold is distributed among the hits in the same time slice. """
    if th <= 0: return hits
    return (hits.groupby("Z", as_index=False)
                .apply(apply_threshold, th=th, on_corrected=on_corrected))
