import numpy  as np
from itertools   import compress
from copy        import deepcopy
from typing      import List
from .. cities.components import split_energy
from .. evm  import event_model as evm

from .. types.ic_types      import NN
from .. types.ic_types      import xy

def merge_NN_hits(hits : List[evm.Hit], same_peak : bool = True) -> List[evm.Hit]: 
    """ Returns a list of the hits where the  energies of NN hits are distributed to the closest hits such that the added energy is proportional to 
    the hit energy. If all the hits were NN the function returns empty list. """
    nn_hits     = [h for h in hits if h.Q==NN]
    non_nn_hits = [deepcopy(h) for h in hits if h.Q!=NN]
    passed = len(non_nn_hits)>0
    if not passed:
        return []
    hits_to_correct=[]
    for nn_h in nn_hits:
        peak_num = nn_h.npeak
        if same_peak:
            hits_to_merge = [h for h in non_nn_hits if h.npeak==peak_num]
        else:
            hits_to_merge = non_nn_hits
        try:
            z_closest  = min(abs(h.Z-nn_h.Z) for h in hits_to_merge)
        except ValueError:
            continue
        h_closest = [h for h in hits_to_merge if abs(h.Z-nn_h.Z)==z_closest]
        en_tot = sum([h.E for h in h_closest])
        for h in h_closest:
            hits_to_correct.append([h,nn_h.E*(h.E/en_tot)])

    for h, en in hits_to_correct:
        h.energy += en
    return non_nn_hits

def threshold_hits(hits : List[evm.Hit], th : float) -> List[evm.Hit]:
    """Returns list of the hits which charge is above the threshold. The energy of the hits below the threshold is distributed among the hits in the same time slice. """
    if th==0:
        return hits
    else:
        new_hits=[]
        for z_slice in np.unique([x.Z for x in hits]):
            slice_hits  = [x for x in hits if x.Z == z_slice]
            e_slice     = sum([x.E for x in slice_hits])
            mask_thresh = np.array([x.Q>=th for x in slice_hits])
            if sum(mask_thresh)<1:
                hit = evm.Hit(slice_hits[0].npeak, evm.Cluster(NN, xy(0,0), xy(0,0), 0), z_slice, e_slice, xy(slice_hits[0].Xpeak,slice_hits[0].Ypeak))
                new_hits.append(hit)
                continue
            hits_pass_th=list(compress(deepcopy(slice_hits), mask_thresh))
            es = split_energy(e_slice, hits_pass_th)
            for i,x in enumerate(hits_pass_th):
                x.energy=es[i]
                new_hits.append(x)
        return new_hits
