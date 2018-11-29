import os
from . esmeralda import merge_NN_hits
from . esmeralda import threshold_hits
import numpy                     as     np
from   numpy.testing             import assert_allclose,assert_equal, assert_almost_equal
from   .. core.configure         import configure
from   .. evm.event_model        import Hit
from   .. evm.event_model        import HitCollection
from   .. evm.event_model        import Cluster
from   .. types.ic_types         import xy
from   .. core.system_of_units_c import units
from   .. types.ic_types         import NN
from   .. reco                   import dst_functions    as dstf
from    . penthesilea            import penthesilea
from   .. io                     import hits_io          as hio
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
    ens  = [h.E for h in hitc.hits if h.Q != NN]
    passed_same_peak, hitc_same_peak = merge_NN_hits(hitc)
    els_test_same_peak = [h.El for h in hitc_same_peak.hits]
    els_same_peak      = np.array([ 1020, 1530, 3000]) * units.pes
    passed,hitc = merge_NN_hits(hitc, same_peak=False)
    els_test = [h.El for h in hitc.hits]
    els      = np.array([ 1060, 1590, 3000]) * units.pes
    #check the El attribute changes appropriately
    assert_allclose(els_test_same_peak  , els_same_peak, rtol=1e-4)
    assert_allclose(els_test  , els, rtol=1e-4)
    #check the E attribute not changed
    assert_equal(ens, [h.E for h in hitc_same_peak.hits])
    assert_equal(ens, [h.E for h in hitc          .hits])
    #check event passes filter
    assert passed_same_peak == True
    assert passed == True
    
def test_threshold_hits(config_tmpdir, Kr_pmaps_run4628_filename):
    PATH_IN =  Kr_pmaps_run4628_filename
    nrequired = 1
    conf = configure('dummy invisible_cities/config/penthesilea.conf'.split())
    PATH_OUT_2 = os.path.join(config_tmpdir, 'KrDST_4628_2th.h5')
    conf.update(dict(run_number = 4628,
                     files_in   = PATH_IN,
                     file_out   = PATH_OUT_2,
                     event_range = (0, nrequired),
                     slice_reco_params = dict(
                         Qthr          = 2 * units.pes,
                         Qlm           = 0 * units.pes,
                         lm_radius     = 0 * units.mm ,
                         new_lm_radius = 0 * units.mm ,
                         msipm         = 1      )))
    cnt = penthesilea (**conf)
    PATH_OUT_10 = os.path.join(config_tmpdir, 'KrDST_4628_10th.h5')
    conf.update(dict(run_number = 4628,
                     files_in   = PATH_IN,
                     file_out   = PATH_OUT_10,
                     event_range = (0, nrequired),
                     slice_reco_params = dict(
                         Qthr          = 10 * units.pes,
                         Qlm           = 0 * units.pes,
                         lm_radius     = 0 * units.mm ,
                         new_lm_radius = 0 * units.mm ,
                         msipm         = 1      )))
    cnt = penthesilea (**conf)
    hits_pent_2  = hio.load_hits(PATH_OUT_2)
    hits_pent_10 = hio.load_hits(PATH_OUT_10)
    ev_num = 1
    hits_thresh  = threshold_hits (hits_pent_2[ev_num], th=10)
    assert len(hits_pent_10[1].hits)==len(hits_thresh.hits)
    for h1, h2 in zip(hits_pent_10[1].hits,hits_thresh.hits):
        assert_almost_equal (h1.E, h2.E, 3)
        assert_almost_equal (h1.Q, h2.Q, 3)
        assert h1.Z==h2.Z
        assert h1.X==h2.X
        assert h1.Y==h2.Y
