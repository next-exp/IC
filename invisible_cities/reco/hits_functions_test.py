import numpy                     as     np

from   .. evm.event_model        import Hit
from   .. evm.event_model        import HitCollection
from   .. evm.event_model        import Cluster
from   .. types.ic_types         import xy
from   .. core.system_of_units_c import units
from   .. types.ic_types         import NN
from   .. reco                   import dst_functions    as dstf
from   .. reco.hits_functions    import merge_NN_hits

from   numpy.testing             import assert_allclose


from pytest import fixture

@fixture
def toy_hitc():
    x_peak,y_peak=1 * units.mm, 1 * units.mm
    xs = np.array([65, 20,0, -64]) * units.mm
    ys = np.array([63, 21,0, -62]) * units.mm
    zs = np.array([25, 25, 20, 60]) * units.mm
    qs = np.array([ 100, 150, NN, 200]) * units.pes
    es = np.array([ 1000,1500, 50, 3000]) * units.pes
    hitc = HitCollection(0, 0)
    for i in range(len(xs)):
        hitc.hits.append(Hit(0, Cluster(qs[i], xy(xs[i],ys[i]), xy(0,0), 0), zs[i], es[i], xy(x_peak,y_peak)))
    return hitc
    
def test_merge_NN_hits(toy_hitc):
    hitc = toy_hitc
    hitc = merge_NN_hits(hitc)
    ecs_test = [h.E for h in hitc.hits]
    ecs      = np.array([ 1020,1530,3000]) * units.pes
    assert_allclose(ecs_test  , ecs, rtol=1e-4)
