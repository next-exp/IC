import numpy  as np
import pandas as pd

from itertools   import compress
from copy        import deepcopy
from typing      import List

from .. evm  import event_model as evm
from .. types.ic_types      import NN
from .. types.ic_types      import xy

EPSILON = np.finfo(np.float64).eps


def e_from_q(qs: np.ndarray, e: float) -> np.ndarray:
    """
    Distribute some energy among the hits according to their charge.

    Parameters
    ----------
    qs: np.ndarray, shape (n,)
        The charge of each hit.

    e_slice: float
        The energy to be shared, typically of a given slice.

    Returns
    -------
    es: np.ndarray, shape (n,)
        The associated hit energy.
    """
    return qs * e / (qs.sum() + EPSILON)


def sipms_above_threshold(xys: np.ndarray, qs: np.ndarray, thr:float, energy: float):
    """
    Finds SiPMs with charge above threshold and returns their position, charge
    and associated energy.

    Parameters
    ----------
    xys: np.ndarray, shape (n,2)
        SiPM positions
    qs: np.ndarray, shape (n,)
        Charge of each SiPM.
    thr: float
        Threshold on SiPM charge.
    energy: float
        Energy to be shared among the hits.

    Returns
    -------
    xs: np.ndarray, shape (m,)
        x positions of the SiPMs above threshold
    ys: np.ndarray, shape (m,)
        y positions of the SiPMs above threshold
    qs: np.ndarray, shape (m,)
        Charge of the SiPMs above threshold
    es: np.ndarray, shape (m,)
        Associated energy of each hit
    """
    over_thr = qs >= thr
    nonempty = np.any(over_thr)

    xy =        xys[over_thr]
    qs =         qs[over_thr] if nonempty else [NN]
    xs =             xy[:, 0] if nonempty else [NN]
    ys =             xy[:, 1] if nonempty else [NN]
    es = e_from_q(qs, energy) if nonempty else [energy]
    return xs, ys, qs, es




def merge_NN_hits(hits: pd.DataFrame, same_peak: bool = True) -> pd.DataFrame:
    """
    Finds NN hits (defined as hits with Q=NN) and removes them without energy
    losses. The energy of each NN hit (both E and Ec) is distributed to the
    closest non-NN hit or hits (if many, they must be at exactly the same
    distance). This is done proportionally to the energy of the receiving non-NN
    hits.

    The definition of closest hit can be tweaked by the `same_peak` argument,
    which determiness if the receiving hits must be in the same S2 peak or can
    be from any S2 within the same event.

    If the input contains only NN hits, the output is empty.

    Parameters
    ----------
    hits: pd.DataFrame
        Input hits. Must include at least the following columns:
        `Q`, `npeak`, `Z`, `E`, and `Ec`.

    same_peak: bool, optional
        If `True`, only hits within the same S2 peak as the `NN` hit are
        considered for merging. If `False`, all hits are considered regardless
        of `npeak`. Default is True.

    Returns
    -------
    merged_hits: pd.DataFrame
        A copy of the input with NN hits removed and energy reassigned.

    Notes
    -----
    - The merging process conserves the total `E` and `Ec` across all hits.
    - If a `NN` hit does not have any neighbours, the `NN` hit is effectively
      dropped.
    """
    sel = hits.Q == NN
    if not np.any(sel): return hits # save some time

    nn_hits = hits.loc[ sel]
    hits    = hits.loc[~sel].copy()

    corrections = pd.DataFrame(dict(E=0, Ec=0), index=hits.index.values)
    for _, nn_hit in nn_hits.iterrows():
        candidates = hits.loc[hits.npeak == nn_hit.npeak] if same_peak else hits
        if len(candidates) == 0: continue # drop hit !!! dangerous

        # find closest hit or hits
        dz      = np.abs(candidates.Z - nn_hit.Z)
        closest = candidates.loc[np.isclose(dz, dz.min())]
        index   = closest.index

        # redistribute energy proportionally to the receiving hits' energy
        # corrections are accumulated to make this process order insentitive
        corr_e  = nn_hit.E  * closest.E  / closest.E .sum()
        corr_ec = nn_hit.Ec * closest.Ec / closest.Ec.sum()
        corrections.loc[index, "E Ec".split()] += np.stack([corr_e, corr_ec], axis=1)

    # apply correction factors based on original charge values
    hits.loc[:, "E Ec".split()] += corrections.values
    return hits


def empty_hit( event : int  , timestamp: float, peak_no: int
             , x_peak: float, y_peak   : float, z      : float
             , e     : float, ec       : float):
    """
    Produces an empty hit with NN x and y coordinates and NN charge.
    Non-tracking data is taken from input.
    """
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


def apply_threshold(hits: pd.DataFrame, th: float, on_corrected: bool = False) -> pd.DataFrame:
    """
    Apply a charge threshold to filter hits and renormalize their energies.

    Input hits with charge (either `Q` or `Qc`) below `th` are removed. The
    energy of the hit collection is preserved and redistributed to the surviving
    hits. If no hits survive the threshold, an empty hit (with NN charge, x and
    y) is returned using the event metadata of the first hit.

    Parameters
    ----------
    hits : pd.DataFrame
        Input hits. All hit columns must be present.

    th : float
        Charge threshold in pe.

    on_corrected : bool, optional
        Whether to use the regular charge `Q` or the corrected charge `Qc`.
        Default is False.

    Returns
    -------
    thresholded_hits: pd.DataFrame
        Hits surviving the threshold, with renormalized `E` and `Ec` values. If
        no hits pass the threshold, an empty hit (with NN charge and x,y
        position) is returned.

    Notes
    -----
    - Energy renormalization ensures that the sum of `E` and `Ec` for the remaining
      hits equals the sum of `E` and `Ec` in the original `hits` DataFrame.
    - If no hits survive the threshold, the returned DataFrame has a single "empty"
      hit corresponding to the first hit's event metadata.
    """
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


def threshold_hits(hits: pd.DataFrame, th: float, on_corrected: bool=False) -> pd.DataFrame:
    """
    Apply a charge threshold (`th`)vto the hits for each Z slice separately. If
    the threshold is negative or zero, the function returns the input DataFrame
    unchanged.

    Parameters
    ----------
    hits : pd.DataFrame
        Input hits. All hit columns must be present.

    th : float
        Charge threshold in pe.

    on_corrected : bool, optional
        Whether to use the regular charge `Q` or the corrected charge `Qc`.
        Default is False.

    Returns
    -------
    thresholded_hits: pd.DataFrame
        Hits surviving the threshold, with renormalized `E` and `Ec` values.
        Slices with no surviving hits produce an empty hit (a.k.a NN hit).

    Notes
    -----
    - Energy renormalization ensures that the sum of `E` and `Ec` for the remaining
      hits equals the sum of `E` and `Ec` in the original `hits` DataFrame.
    - If no hits survive the threshold, the returned DataFrame has a single "empty"
      hit corresponding to the first hit's event metadata.
    - See `apply_threshold` for further details.
    """
    if th <= 0: return hits
    return (hits.groupby("Z", as_index=False)
                .apply(apply_threshold, th=th, on_corrected=on_corrected))
