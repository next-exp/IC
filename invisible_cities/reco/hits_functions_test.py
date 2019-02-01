import os
import numpy                     as     np
from   numpy.testing             import assert_allclose
from   numpy.testing             import assert_almost_equal
from   .. core.configure         import configure
from   .. evm.event_model        import Hit
from   .. evm.event_model        import Cluster
from   .. types.ic_types         import xy
from   .. core.system_of_units_c import units
from   .. types.ic_types         import NN
from   .. cities.penthesilea     import penthesilea
from   .. io                     import hits_io          as hio
from   .  hits_functions         import merge_NN_hits
from   .  hits_functions         import threshold_hits
from  pytest                     import fixture
from  pytest                     import mark
from hypothesis                  import given
from hypothesis                  import settings
from hypothesis.strategies       import lists
from hypothesis.strategies       import floats
from hypothesis.strategies       import integers
from copy                        import deepcopy
from hypothesis                  import assume
from hypothesis.strategies       import composite

@composite
def hit(draw, min_value=1, max_value=100):
    x     = draw(floats  (  1,   5))
    y     = draw(floats  (-10,  10))
    xvar  = draw(floats  (.01,  .5))
    yvar  = draw(floats  (.10,  .9))
    Q     = draw(floats  ( -10, 100).map(lambda x: NN if x<=0 else x))
    nsipm = draw(integers(  1,  20))
    npeak = 0
    z     = draw(floats  ( 50, 100))
    E     = draw(floats  ( 50, 100))
    x_peak= draw(floats  (  1,   5))
    y_peak= draw(floats  (-10,  10))
    return Hit(npeak,Cluster(Q,xy(x,y),xy(xvar,yvar),nsipm),z,E,xy(x_peak,y_peak))

@composite
def list_of_hits(draw):
    list_of_hits = draw(lists(hit(), min_size=2, max_size=10))
    assume(sum((h.Q >0 for h in list_of_hits))>=1)
    return list_of_hits
@given(list_of_hits())
def test_merge_NN_not_modify_input(hits):
    hits_org=deepcopy(hits)
    before_len = len(hits)
    hits_merged = merge_NN_hits(hits)
    after_len = len(hits)
    assert before_len == after_len
    for h1, h2 in zip(hits, hits_org):
        assert h1== h2

@given(list_of_hits())
def test_merge_hits_energy_conserved(hits):
    hits_merged = merge_NN_hits(hits)
    assert_almost_equal(sum((h.E for h in hits)), sum((h.E for h in hits_merged)))

def test_merge_NN_hits_exact(TlMC_hits, TlMC_hits_merged):
    for ev, hitc in TlMC_hits.items():
        hits_test  = TlMC_hits_merged[ev].hits
        hits_merged = merge_NN_hits(hitc.hits)
        assert len(hits_test)==len(hits_merged)
        for h1, h2 in zip(hits_test,hits_merged):
            print(ev,'\n','hi\n',h1,'\n','h2\n',h2)
            assert h1==h2
