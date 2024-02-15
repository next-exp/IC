import sys
import numpy  as np
import pandas as pd

from .. core.core_functions  import weighted_mean_and_var
from .. core                 import system_of_units as units
from .. core.exceptions      import SipmEmptyList
from .. core.exceptions      import SipmZeroCharge
from .. core.exceptions      import SipmEmptyListAboveQthr
from .. core.exceptions      import SipmZeroChargeAboveQthr
from .. core.exceptions      import ClusterEmptyList
from .. core.configure       import check_annotations

from .. types.ic_types       import xy
from .. evm.event_model      import Cluster

from typing import Optional
from typing import Sequence
from typing import Tuple


def threshold_check( pos : np.ndarray # (n, 2)
                   ,  qs : np.ndarray # (n,)
                   , thr : float) -> Tuple[np.ndarray, np.ndarray]:
    if not len(pos)   : raise SipmEmptyList
    if np.sum(qs) == 0: raise SipmZeroCharge

    above_threshold = np.where(qs >= thr)[0]
    pos, qs = pos[above_threshold], qs[above_threshold]

    if not len(pos)   : raise SipmEmptyListAboveQthr
    if np.sum(qs) == 0: raise SipmZeroChargeAboveQthr

    return pos, qs


@check_annotations
def barycenter( pos : np.ndarray # (n, 2)
              , qs  : np.ndarray # (n,)
              , Qthr: Optional[float] = 0 * units.pes):
    """pos = column np.array --> (matrix n x 2)
       ([x1, y1],
        [x2, y2]
        ...
        [xs, ys])
       qs = vector (q1, q2...qs) --> (1xn)
    """
    pos, qs = threshold_check(pos, qs, Qthr)
    mu, var = weighted_mean_and_var(pos, qs, axis=0)
    # For uniformity of interface, all xy algorithms should return a
    # list of clusters. barycenter always returns a single clusters,
    # but we still want it in a list.
    return [Cluster(np.sum(qs), xy(*mu), xy(*var), len(qs))]


def discard_sipms(sis, pos, qs):
    return np.delete(pos, sis, axis=0), np.delete(qs, sis)


def get_nearby_sipm_inds(cs, d, pos):
    """return indices of sipms less than d from (xc,yc)"""
    return np.where(np.linalg.norm(pos - cs, axis=1) <= d)[0]


def count_masked(cs, d, datasipm, is_masked):
    if is_masked is None: return 0

    pos = np.stack([datasipm.X.values, datasipm.Y.values], axis=1)
    indices = get_nearby_sipm_inds(cs, d, pos)
    return np.count_nonzero(~is_masked.astype(bool)[indices])


@check_annotations
def corona( pos             : np.ndarray # (n, 2)
          , qs              : np.ndarray # (n,)
          , all_sipms       : pd.DataFrame
          , Qthr            : float
          , Qlm             : float
          , lm_radius       : float
          , new_lm_radius   : float
          , msipm           : int
          , consider_masked : Optional[bool] = False) -> Sequence[Cluster]:
    """
    Creates a list of clusters with the following steps:
    - identifying the SiPM with highest charge (which must be > `Qlm`)
    - obtaining the barycenter from the SiPMs within `lm_radius` of
      the SiPM with highest charge to locate the local maximum of the
      charge distribution
    - obtaining the barycenter from the SiPMs within `new_lm_radius`
      of the new local maximum
    - creating a cluster from the SiPMs neighbouring the barycenter of
      the new local maximum, provided it contains at least `msipm`
    - discarding (non-destructively) the sipms that formed the cluster
    - repeating the previous steps until there are no more SiPMs with
      charge higher than `Qlm`

    Parameters
    ----------
    pos : np.ndarray with shape (nx2)
        SiPM positions. The first and second columns corresponds to
        the x and y positions, respectively

    qs : np.ndarray with shape (n,)
        SiPM charges

    Qthr : float
        Charge threshold to apply to all SiPMs. SiPMs with lower
        charge are ignored

    Qlm : float
        Charge threshold to find a local maximum

    msipm : int
        Minimum number of SiPMs in a cluster (see `consider_masked`)

    pitch : float
        Distance between contiguous SiPMs in each axis

    lm_radius : float
        Distance from the SiPM with highest charge with which a new
        local maximum is estimated (see Notes)

    new_lm_radius : float
        Radius used for the calculation of the barycenter from the new
        local maximum (see Notes)

    consider_masked : bool
        Whether to consider masked SiPMs in the clustering
        algorithm. It affects particularly `msipm`. If `True`,
        clusters might contain less than `msipm` SiPMs, if any of
        those is a masked sensor.

    Returns
    -------
    clusters : List[Cluster]
        The list of clusters based on the SiPM pattern

    Notes
    -----
    The algorithm follows the following logic:
    Find a new local maximum by taking the barycenter of the SiPMs
    within a distance `lm_radius` of the SiPM with highest
    charge. ***In general `lm_radius` should typically be set to 0, or
    some value slightly larger than pitch, such as pitch * sqrt(2).***
    This kwarg has some physical motivation. It exists to try to
    partially compensate the problem that the tracking plane is not
    continuous even though light can be emitted by the EL from any
    (x,y). When `lm_radius` < pitch, the search for SiPMs that might
    contribute pos and charge to a new cluster is always centered
    about the position of hottest sipm. In contrast, when `lm_radius
    >= pitch`, the search for SiPMs contributing to the new cluster
    can be centered at any (x,y). Consider the case where there are
    four SiPMs with nearly equal charge around a local maximum. The
    new local maximum would yield a position, pos1, somewhere among
    these SiPMs. Searching for SiPMs that contribute to this cluster
    within `new_lm_radius` from pos1 might be better than searching
    for SiPMs within `new_lm_radius` from the SiPM with highest
    charge. We should be aware that setting `lm_radius` to some
    distance greater than the pitch, we allow `new_local_maximum` to
    assume any (x,y) but we also create the effect that depending on
    where `new_local_maximum` is, more or fewer SiPMs will be within
    `new_lm_radius`. This effect does not exist when `lm_radius` is 0.
    . Find a new cluster by calling barycenter on SiPMs within
    `new_lm_radius` of the new local maximum (see `lm_radius`).

    Example
    -------

    # In order to create Clusters in NEW from 3x3 blocks of SiPMs with
    # a minimum of 3 SiPMs and 5 pes for the SiPM with highest charge
    # in each cluster, one could call:
    corona(pos, qs, all_sipms,
           Qthr            =  1 * units.pes,
           Qlm             =  5 * units.pes,
           lm_radius       =  15 * units.mm,
           new_lm_radius   =  15 * units.mm, # slightly longer than the pitch
           msipm           =  3,
           consider_masked = True)

    """
    assert     lm_radius >= 0,     "lm_radius must be non-negative"
    assert new_lm_radius >= 0, "new_lm_radius must be non-negative"

    pos, qs = threshold_check(pos, qs, Qthr)
    masked  = all_sipms.Active.values.astype(bool) if consider_masked else None

    c  = []
    # While there are more local maxima
    while len(qs) > 0:

        hottest_sipm = np.argmax(qs)       # SiPM with largest Q
        if qs[hottest_sipm] < Qlm: break   # largest Q remaining is negligible

        # find new local maximum of charge considering all SiPMs within lm_radius of hottest_sipm
        within_lm_radius  = get_nearby_sipm_inds(pos[hottest_sipm], lm_radius, pos)
        new_local_maximum = barycenter(pos[within_lm_radius], qs[within_lm_radius])[0].posxy

        # find the SiPMs within new_lm_radius of the new local maximum of charge
        within_new_lm_radius = get_nearby_sipm_inds(new_local_maximum, new_lm_radius, pos      )
        n_masked_neighbours  = count_masked        (new_local_maximum, new_lm_radius, all_sipms, masked)

        # if there are at least msipms within_new_lm_radius, taking
        # into account any masked channel, get the barycenter
        if len(within_new_lm_radius) >= msipm - n_masked_neighbours:
            c.extend(barycenter(pos[within_new_lm_radius], qs[within_new_lm_radius]))
            # delete the SiPMs contributing to this cluster

        pos, qs = discard_sipms(within_new_lm_radius, pos, qs)

    if not len(c): raise ClusterEmptyList

    return c
