import os

from pytest        import mark
from numpy.testing import assert_almost_equal

from   .. core.configure       import configure
from   .. evm.event_model      import Hit
from   .. evm.event_model      import Cluster
from   .. types.ic_types       import xy
from   .. core                 import system_of_units as units
from   .. core.testing_utils   import assert_hit_equality
from   .. types.ic_types       import NN
from   .. cities.penthesilea   import penthesilea
from   .. io                   import hits_io          as hio
from   .  hits_functions       import merge_NN_hits
from   .  hits_functions       import threshold_hits
from hypothesis                import given
from hypothesis                import settings
from hypothesis.strategies     import lists
from hypothesis.strategies     import floats
from hypothesis.strategies     import integers
from copy                      import deepcopy
from hypothesis                import assume
from hypothesis.strategies     import composite

@composite
def hit(draw, min_value=1, max_value=100):
    x      = draw(floats  (  1,   5))
    y      = draw(floats  (-10,  10))
    xvar   = draw(floats  (.01,  .5))
    yvar   = draw(floats  (.10,  .9))
    Q      = draw(floats  (-10, 100).map(lambda x: NN if x<=0 else x))
    nsipm  = draw(integers(  1,  20))
    npeak  = 0
    z      = draw(floats  ( 50, 100))
    E      = draw(floats  ( 50, 100))
    Ec     = draw(floats  ( 50, 100))
    x_peak = draw(floats  (  1,   5))
    y_peak = draw(floats  (-10,  10))
    Qc     = draw(floats  (  0, 100).map(lambda x: -1 if Q==NN else x))
    assume(abs(Qc - Q) > 1e-3)
    return Hit(npeak,Cluster(Q, xy(x, y), xy(xvar, yvar), nsipm,  Qc=Qc), z, E, xy(x_peak, y_peak), s2_energy_c=Ec)

@composite
def list_of_hits(draw):
    list_of_hits = draw(lists(hit(), min_size=2, max_size=10))
    assume(sum((h.Q > 0 for h in list_of_hits if h.Q != NN)) >= 1)
    return list_of_hits

@composite
def thresholds(draw, min_value=1, max_value=1):
    th1 = draw (integers(  10   ,  20))
    th2 = draw (integers(  th1+1,  30))
    return th1, th2

@given(list_of_hits())
def test_merge_NN_does_not_modify_input(hits):
    hits_org    = deepcopy(hits)
    before_len  = len(hits)

    merge_NN_hits(hits)

    after_len   = len(hits)

    assert before_len == after_len
    for h1, h2 in zip(hits, hits_org):
        assert_hit_equality(h1, h2)

@given(list_of_hits())
def test_merge_hits_energy_conserved(hits):
    hits_merged = merge_NN_hits(hits)
    assert_almost_equal(sum((h.E  for h in hits)), sum((h.E  for h in hits_merged)))
    assert_almost_equal(sum((h.Ec for h in hits)), sum((h.Ec for h in hits_merged)))


@given(list_of_hits(), floats())
def test_threshold_hits_does_not_modify_input(hits, th):
    hits_org    = deepcopy(hits)
    before_len  = len(hits)

    threshold_hits(hits,th)

    after_len   = len(hits)
    assert before_len == after_len
    for h1, h2 in zip(hits, hits_org):
        assert_hit_equality(h1, h2)


@mark.parametrize("on_corrected", (False, True))
@given(hits=list_of_hits(), th=floats())
def test_threshold_hits_energy_conserved(hits, th, on_corrected):
    hits_thresh = threshold_hits(hits, th, on_corrected=on_corrected)
    assert_almost_equal(sum((h.E  for h in hits)), sum((h.E  for h in hits_thresh)))
    assert_almost_equal(sum((h.Ec for h in hits)), sum((h.Ec for h in hits_thresh)))


@mark.parametrize("on_corrected", (False, True))
@given(hits=list_of_hits(), th=floats())
def test_threshold_hits_all_larger_than_th(hits, th, on_corrected):
    hits_thresh  = threshold_hits(hits, th, on_corrected = on_corrected)
    assert (h.Q  > th or h.Q  == NN for h in hits_thresh)
    assert (h.Qc > th or h.Qc == NN for h in hits_thresh)
