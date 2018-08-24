import numpy                     as     np
from   numpy.testing             import assert_allclose

from   .. evm.event_model        import Hit
from   .. evm.event_model        import HitCollection
from   .. evm.event_model        import Cluster
from   .. types.ic_types         import xy
from   .. core.system_of_units_c import units
from   .. types.ic_types         import NN
from   .. reco                   import dst_functions    as dstf
from   .. reco.hits_functions    import merge_NN_hits
from   .. reco.hits_functions    import hitc_corrections

from  pytest                     import fixture

@fixture
def toy_hitc():
    x_peaks =np.array([1, 1, 1, 1, 0]) * units.mm
    y_peaks =np.array([1, 1, 1, 1, 0]) * units.mm
    xs = np.array([65, 20,0, -64, 0]) * units.mm
    ys = np.array([63, 21,0, -62, 0]) * units.mm
    zs = np.array([25, 25, 20, 60, 25]) * units.mm
    qs = np.array([ 100, 150, NN, 200,NN]) * units.pes
    es = np.array([ 1000,1500, 50, 3000,100]) * units.pes
    ns = np.array([0,0,0,0,1])
    hitc = HitCollection(0, 0)
    for i in range(len(xs)):
        hitc.hits.append(Hit(ns[i], Cluster(qs[i], xy(xs[i],ys[i]), xy(0,0), 1), zs[i], es[i], xy(x_peaks[i],y_peaks[i])))
    return hitc
@fixture
def toy_emap():
    raw_e = [1000.     , 1500.     , 3000.     ]
    lt_e  = [1002.45422, 1503.68134, 3017.80762]
    cor_e = [2948.58558, 4422.87837, 8929.88863]
    raw_q = [ 100.     ,  150.     ,  200.     ]
    lt_q  = [ 100.24542,  150.36813,  201.18717]
    cor_q = [ 294.85855,  442.28783,  595.32590]
    return raw_e, lt_e, cor_e, raw_q, lt_q, cor_q

def test_merge_NN_hits(toy_hitc):
    hitc = toy_hitc
    hitc_same_peak = merge_NN_hits(hitc)
    ecs_test_same_peak = [h.E for h in hitc_same_peak.hits]
    ecs_same_peak     = np.array([ 1020,1530,3000]) * units.pes
    hitc = merge_NN_hits(hitc, same_peak=False)
    ecs_test = [h.E for h in hitc.hits]
    ecs     = np.array([ 1060,1590,3000]) * units.pes
    assert_allclose(ecs_test_same_peak  , ecs_same_peak, rtol=1e-4)
    assert_allclose(ecs_test  , ecs, rtol=1e-4)

def test_hitc_corrections(toy_hitc, toy_emap, corr_toy_data):
    corr_filename, _ = corr_toy_data
    XYcor    = dstf.load_xy_corrections         (corr_filename, group = "Corrections", node =  "XYcorrections", norm_strategy = 'max')
    LTcor    = dstf.load_lifetime_xy_corrections(corr_filename, group = "Corrections", node  = "LifetimeXY"                         )
    hitc_emap= hitc_corrections(toy_hitc, XYcor, LTcor, XYcor, LTcor)
    raw_e, lt_e, cor_e, raw_q, lt_q, cor_q = toy_emap
    assert_allclose(raw_e , hitc_emap.E.raw, rtol=1e-4)
    assert_allclose( lt_e , hitc_emap.E.lt , rtol=1e-4)
    assert_allclose(cor_e , hitc_emap.E.cor, rtol=1e-4)
    assert_allclose(raw_q , hitc_emap.Q.raw, rtol=1e-4)
    assert_allclose( lt_q , hitc_emap.Q.lt , rtol=1e-4)
    assert_allclose(cor_q , hitc_emap.Q.cor, rtol=1e-4)

