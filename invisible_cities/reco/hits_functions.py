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
