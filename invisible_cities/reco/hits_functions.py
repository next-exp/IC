import  numpy               as np
from    copy            import copy
from .. evm.event_model import Hit
from .. evm.event_model import HitCollection

from .. types.ic_types  import NN

from typing             import NamedTuple

class EVector(NamedTuple):
    raw : float            # raw value
    lt  : float            # corrected by lifetime
    cor : float            # fully corrected
                                    
class HitEnergyMap(NamedTuple):
    E   :  EVector
    Q   :  EVector

def merge_NN_hits(hitc, same_peak=True):
    """ Returns a modified HitCollection instance by adding energies of NN hits to closest
    hits such that the added energy is proportional to the hit energy. If all the hits were NN
    the function returns empty HitCollection."""
    hitc_new    = HitCollection(hitc.event, hitc.time)
    nn_hits     = [h for h in hitc.hits if h.Q==NN]
    non_nn_hits = [h for h in hitc.hits if h.Q!=NN]
    if len(non_nn_hits)==0:
        return hitc_new
    for h in non_nn_hits:
        hitc_new.hits.append(copy(h))
    for nn_h in nn_hits:
        peak_num=nn_h.npeak
        if same_peak:
            hits_to_merge = [h for h in hitc_new.hits if h.npeak==peak_num]
        else:
            hits_to_merge = hitc_new.hits
        try:
            z_closest  = min(hits_to_merge , key=lambda h: np.abs(h.Z-nn_h.Z)).Z
        except ValueError:
            continue
        h_closest      = [h for h in hits_to_merge if h.Z==z_closest]
        h_closest_etot = sum([h.E for h in h_closest])
        for h in h_closest:
            h.energy += nn_h.E*(h.E/h_closest_etot)
    return hitc_new

def hitc_corrections(hitc, EXYcor, ELTcor, QXYcor, QLTcor, ftlife=1):
    """Returns HitEnergyMap of corrected HitCollection skipping NN hits."""
    x_hits = [hit.X for hit in hitc.hits if hit.Q !=NN]
    y_hits = [hit.Y for hit in hitc.hits if hit.Q !=NN]
    z_hits = [hit.Z for hit in hitc.hits if hit.Q !=NN]
    e_hits = [hit.E for hit in hitc.hits if hit.Q !=NN]
    q_hits = [hit.Q for hit in hitc.hits if hit.Q !=NN]

    qlt    = q_hits  * QLTcor(z_hits, x_hits, y_hits).value**(ftlife)
    qcor   = qlt     * QXYcor(x_hits,y_hits)         .value
    elt    = e_hits  * ELTcor(z_hits, x_hits, y_hits).value**(ftlife)
    ecor   = elt     * EXYcor(x_hits,y_hits)         .value
    
    Evec   = EVector (np.array(e_hits), elt, ecor)
    Qvec   = EVector (np.array(q_hits), qlt, qcor)
    
    return HitEnergyMap(Evec, Qvec)
