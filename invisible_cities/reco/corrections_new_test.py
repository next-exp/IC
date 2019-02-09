import numpy  as np
from . corrections_new import maps_coefficient_getter
from . corrections_new import read_maps
from . corrections_new import CorrectionsDF
from . corrections_new import ASectorMap
from pytest                import fixture
from numpy.testing         import assert_allclose

@fixture
def toy_corrections():
    xs,ys = np.meshgrid(np.linspace(-199,199,5),np.linspace(-199,199,5))
    xs=xs.flatten(); ys=ys.flatten()
    zs = np.ones(25)
    es = np.ones(25)
    e0_corrections = np.array([9.43322064e-05, 9.43322064e-05, 9.43322064e-05, 9.43322064e-05,
                               9.43322064e-05, 9.43322064e-05, 8.39198694e-05, 8.32223502e-05,
                               9.13077163e-05, 9.43322064e-05, 9.43322064e-05, 8.32826455e-05,
                               8.02149510e-05, 8.70309002e-05, 9.43322064e-05, 9.43322064e-05,
                               9.20405803e-05, 8.84955546e-05, 9.56684281e-05, 9.43322064e-05,
                               9.43322064e-05, 9.43322064e-05, 9.43322064e-05, 9.43322064e-05,
                               9.43322064e-05])
    lt_corrections = np.array([1.0002771 , 1.0002771 , 1.0002771 , 1.0002771 , 1.0002771 ,
                               1.0002771 , 1.00023523, 1.0002237 , 1.00026303, 1.0002771 ,
                               1.0002771 , 1.00024245, 1.00027317, 1.00027545, 1.0002771 ,
                               1.0002771 , 1.00029724, 1.00029364, 1.0002934 , 1.0002771 ,
                               1.0002771 , 1.0002771 , 1.0002771 , 1.0002771 , 1.0002771 ])
    xy_geo_matrix  = np.array([10600.83335512, 10600.83335512, 10600.83335512, 10600.83335512,
                               10600.83335512, 10600.83335512, 11916.12912023, 12016.00288243,
                               10951.97690475, 10600.83335512, 10600.83335512, 12007.30349264,
                               12466.50390868, 11490.17185802, 10600.83335512, 10600.83335512,
                               10864.77287088, 11300.00262915, 10452.76921306, 10600.83335512,
                               10600.83335512, 10600.83335512, 10600.83335512, 10600.83335512,
                               10600.83335512])
    xy_lt_matrix   = np.array([3609.30708684, 3609.30708684, 3609.30708684, 3609.30708684,
                               3609.30708684, 3609.30708684, 4251.67968216, 4470.78086144,
                               3802.38482442, 3609.30708684, 3609.30708684, 4124.99577806,
                               3661.21394636, 3630.89659737, 3609.30708684, 3609.30708684,
                               3364.7524234 , 3405.97343322, 3408.77553025, 3609.30708684,
                               3609.30708684, 3609.30708684, 3609.30708684, 3609.30708684,
                               3609.30708684])
    return xs, ys, zs, es, e0_corrections, lt_corrections,xy_geo_matrix, xy_lt_matrix

def test_maps_coefficient_getter_exact(toy_corrections, correction_map_filename):
    maps=read_maps(correction_map_filename)
    xs, ys, zs, es, _, _, coef_geo, coef_lt = toy_corrections
    get_maps_coefficient_e0= maps_coefficient_getter(maps, CorrectionsDF.e0)
    CE  = get_maps_coefficient_e0(xs,ys)
    get_maps_coefficient_lt= maps_coefficient_getter(maps, CorrectionsDF.lt)
    LT  = get_maps_coefficient_lt(xs,ys)
    assert_allclose (CE, coef_geo)
    assert_allclose (LT, coef_lt)
