from . esmeralda import merge_NN_hits
import numpy                     as     np
from   numpy.testing             import assert_allclose,assert_equal 

from   .. evm.event_model        import Hit
from   .. evm.event_model        import HitCollection
from   .. evm.event_model        import Cluster
from   .. types.ic_types         import xy
from   .. core.system_of_units_c import units
from   .. types.ic_types         import NN
from   .. reco                   import dst_functions    as dstf

from  pytest                     import fixture

@fixture
def toy_hitc():
    x_peaks =np.array([1, 1, 1, 1, 0]) * units.mm
    y_peaks =np.array([1, 1, 1, 1, 0]) * units.mm
    xs = np.array([65, 20,0, -64, 0]) * units.mm
    ys = np.array([63, 21,0, -62, 0]) * units.mm
    zs = np.array([25, 25, 20, 60, 25]) * units.mm
    qs = np.array([ 100,   150, NN,  200,  NN]) * units.pes
    es = np.array([ 1000, 1500, 50, 3000, 100]) * units.pes
    ns = np.array([0,0,0,0,1])
    hitc = HitCollection(0, 0)
    for i in range(len(xs)):
        hitc.hits.append(Hit(ns[i], Cluster(qs[i], xy(xs[i],ys[i]), xy(0,0), 1), zs[i], es[i], xy(x_peaks[i],y_peaks[i])))
    return hitc

def test_merge_NN_hits(toy_hitc):
    hitc = toy_hitc
    ens  = [h.E for h in hitc.hits]
    passed_same_peak, hitc_same_peak = merge_NN_hits(hitc)
    els_test_same_peak = [h.El for h in hitc_same_peak.hits]
    els_same_peak      = np.array([ 1020, 1530, 0, 3000, 0]) * units.pes
    passed,hitc = merge_NN_hits(hitc, same_peak=False)
    els_test = [h.El for h in hitc.hits]
    els      = np.array([ 1060, 1590, 0, 3000, 0]) * units.pes
    #check the El attribute changes appropriately
    assert_allclose(els_test_same_peak  , els_same_peak, rtol=1e-4)
    assert_allclose(els_test  , els, rtol=1e-4)
    #check the E attribute not changed
    assert_equal(ens, [h.E for h in hitc_same_peak.hits])
    assert_equal(ens, [h.E for h in hitc          .hits])
    #check event passes filter
    assert passed_same_peak == True
    assert passed == True
    

