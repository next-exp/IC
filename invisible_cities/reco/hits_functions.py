import  numpy               as np
from    copy            import copy
from .. evm.event_model import Hit
from .. evm.event_model import HitCollection

from .. types.ic_types  import NN

def merge_NN_hits(hitc):
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
        z_closest      = min(hitc_new.hits , key=lambda h: np.abs(h.Z-nn_h.Z)).Z
        h_closest      = [h for h in hitc_new.hits if h.Z==z_closest]
        h_closest_etot = sum([h.E for h in h_closest])
        for h in h_closest:
            h.energy += nn_h.E*(h.E/h_closest_etot)
    return hitc_new
