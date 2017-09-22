cimport numpy as np
import  numpy as np

from .. evm.event_model         import Hit
from .. evm.event_model_c       import Cluster
from .. types.ic_types          import NN
from .. types.ic_types          import xy
from .. core.exceptions         import XYRecoFail
from .. core.system_of_units_c  import units


cpdef collect_hits(hitc, s2si, compute_xy_position, split_energy, double s1_t, double drift_v):
    """
    here hits are computed for each peak and each slice.
    In case of an exception, a hit is still created with a NN cluster.
    (NN cluster is a cluster where the energy is an IC not number NN)
    this allows to keep track of the energy associated to non reonstructed hits.
    """
    cdef int peak_no, slice_no, j, k
    cdef double [:] t_peak, e_peak, es
    cdef double t_lice, e_lice, z

    for peak_no, (t_peak, e_peak) in sorted(s2si.s2d.items()):
        for slice_no in range(len(t_slice)):
            t_slice = t_peak[slice_no]
            e_slice = e_peak[slice_no]
            z = (t_slice - s1_t) * units.ns * drift_v
            try:
                clusters = compute_xy_position(s2si.s2sid[peak_no], slice_no)
                es       = split_energy(e_slice, clusters)
                for k in range(len(clusters)):
                    c   = clusters[k]
                    e   = es[k]
                    hit = Hit(peak_no, c, z, e)
                    hitc.hits.append(hit)
            except XYRecoFail:
                c = Cluster(NN, xy(0,0), xy(0,0), 0)
                hit       = Hit(peak_no, c, z, e_slice)
                hitc.hits.append(hit)

    return hitc
