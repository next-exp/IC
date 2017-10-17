
"""
here hits are computed for each peak and each slice.
In case of an exception, a hit is still created with a NN cluster.
(NN cluster is a cluster where the energy is an IC not number NN)
this allows to keep track of the energy associated to non reonstructed hits.
"""
cpdef collect_hits(hitc, s2si, compute_xy_position, split_energy, double s1_t, double drift_v)
