import numpy  as np
from itertools   import compress
from copy        import deepcopy
from typing      import List
from .. evm  import event_model as evm
from .. types.ic_types      import NN
from .. types.ic_types      import xy

def split_energy(total_e, clusters):
    if len(clusters) == 1:
        return [total_e]
    qs = np.array([c.Q for c in clusters])
    return total_e * qs / np.sum(qs)

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
        h_closest = [h for h in hits_to_merge if np.isclose(abs(h.Z-nn_h.Z), z_closest)]

        total_raw_energy = sum(h.E  for h in h_closest)
        total_cor_energy = sum(h.Ec for h in h_closest)
        for h in h_closest:
            hits_to_correct.append([h, nn_h.E * h.E / total_raw_energy, nn_h.Ec * h.Ec / total_cor_energy])

    for h, raw_e, cor_e in hits_to_correct:
        h.E  += raw_e
        h.Ec += cor_e

    return non_nn_hits

def threshold_hits(hits : List[evm.Hit], th : float, on_corrected : bool=False) -> List[evm.Hit]:
    """Returns list of the hits which charge is above the threshold. The energy of the hits below the threshold is distributed among the hits in the same time slice. """
    if th==0:
        return hits
    else:
        new_hits=[]
        for z_slice in np.unique([x.Z for x in hits]):
            slice_hits  = [x for x in hits if x.Z == z_slice]
            raw_es      = np.array([x.E  for x in slice_hits])
            cor_es      = np.array([x.Ec for x in slice_hits])
            raw_e_slice = np.   sum(raw_es)
            cor_e_slice = np.nansum(cor_es) + np.finfo(np.float64).eps

            if on_corrected:
                mask_thresh = np.array([x.Qc >= th for x in slice_hits])
            else:
                mask_thresh = np.array([x.Q  >= th for x in slice_hits])
            if sum(mask_thresh) < 1:
                hit = evm.Hit( slice_hits[0].npeak
                             , evm.Cluster.empty()
                             , z_slice
                             , raw_e_slice
                             , xy(slice_hits[0].Xpeak, slice_hits[0].Ypeak)
                             , s2_energy_c = cor_e_slice)
                new_hits.append(hit)
                continue
            hits_pass_th = list(compress(deepcopy(slice_hits), mask_thresh))

            raw_es_new = split_energy(raw_e_slice, hits_pass_th)
            cor_es_new = split_energy(cor_e_slice, hits_pass_th)
            for hit, raw_e, cor_e in zip(hits_pass_th, raw_es_new, cor_es_new):
                hit.E  = raw_e
                hit.Ec = cor_e
                new_hits.append(hit)
        return new_hits
