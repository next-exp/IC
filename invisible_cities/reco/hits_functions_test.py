import os

import numpy  as np
import pandas as pd

from pytest        import mark
from numpy.testing import assert_almost_equal

from   .. core.configure       import configure
from   .. evm.event_model      import Hit
from   .. evm.event_model      import Cluster
from   .. types.ic_types       import xy
from   .. core                 import system_of_units as units
from   .. core.testing_utils   import assert_hit_equality
from   .. core.testing_utils   import assert_dataframes_close
from   .. types.ic_types       import NN
from   .. cities.penthesilea   import penthesilea
from   .. io                   import hits_io          as hio
from   .  hits_functions       import e_from_q
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


event_numbers = integers(0, np.iinfo(np.int32).max)

@composite
def hit(draw, event=None):
    event = draw(event_numbers) if event is None else event
    Q     = draw(floats  (-10, 100).map(lambda x: NN if x<=0  else x).filter(lambda x: abs(x)>.1))
    Qc    = draw(floats  (  0, 100).map(lambda x: -1 if Q==NN else x).filter(lambda x: abs(x)>.1))
    assume(abs(Qc - Q) > 1e-3)

    hit = pd.DataFrame(dict( event    = event
                           , time     = 0
                           , npeak    = 0
                           , Xpeak    = draw(floats  (  1,   5))
                           , Ypeak    = draw(floats  (-10,  10))
                           , nsipm    = draw(integers(  1,  20))
                           , X        = draw(floats  (  1,   5))
                           , Y        = draw(floats  (-10,  10))
                           , Xrms     = draw(floats  (.01,  .5))
                           , Yrms     = draw(floats  (.10,  .9))
                           , Z        = draw(floats  ( 50, 100))
                           , Q        = Q
                           , E        = draw(floats  ( 50, 100))
                           , Qc       = Qc
                           , Ec       = draw(floats  ( 50, 100))
                           , track_id = -1
                           , Ep       = -1
                            ), index=[0])
    return hit


@composite
def list_of_hits(draw):
    event  = draw(event_numbers)
    hits   = draw(lists(hit(event), min_size=2, max_size=10))
    hits   = pd.concat(hits, ignore_index=True)
    non_nn = hits.Q[hits.Q != NN]
    assume(non_nn.sum() >  0)
    assume(non_nn.size  >= 1)
    return hits


@composite
def thresholds(draw, min_value=1, max_value=1):
    th1 = draw (integers(  10   ,  20))
    th2 = draw (integers(  th1+1,  30))
    return th1, th2


def test_e_from_q_simple():
    e  = 1
    qs = np.linspace(12, 34, 56)
    s  = qs.sum()
    es = e_from_q(qs, e)
    assert_almost_equal(es, qs/s)

def test_e_from_q_uniform():
    qs = np.ones(12)
    e  = 345
    es = e_from_q(qs, e)
    assert_almost_equal(es[0], es)

@given(lists(floats(1, 10), min_size=1, max_size=20))
def test_e_from_q_conserves_energy(qs):
    qs = np.asarray(qs)
    e  = 5678
    es = e_from_q(qs, e)
    assert np.isclose(es.sum(), e)

def test_e_from_q_does_not_crash_with_empty_input():
    empty  = np.array([])
    output = e_from_q(empty, 1234)
    assert_almost_equal(output, empty)

def test_e_from_q_does_not_crash_with_zeros():
    zeros  = np.zeros(12)
    output = e_from_q(zeros, 1234)
    assert_almost_equal(output, zeros)

@given(list_of_hits())
def test_merge_NN_does_not_modify_input(hits):
    hits_org = deepcopy(hits)
    merge_NN_hits(hits)
    assert_dataframes_close(hits_org, hits)


@given(list_of_hits())
def test_merge_hits_energy_conserved(hits):
    hits_merged = merge_NN_hits(hits)
    assert_almost_equal(hits.E .sum(), hits_merged.E .sum())
    assert_almost_equal(hits.Ec.sum(), hits_merged.Ec.sum())


@given(list_of_hits())
def test_merge_nn_hits_does_not_leave_nn_hits(hits):
    hits_merged = merge_NN_hits(hits)
    assert all(hits_merged.Q != NN)


@given(list_of_hits(), floats())
def test_threshold_hits_does_not_modify_input(hits, th):
    hits_org = deepcopy(hits)
    threshold_hits(hits, th)
    assert_dataframes_close(hits_org, hits)

@mark.parametrize("on_corrected", (False, True))
@given(hits=list_of_hits(), th=floats())
def test_threshold_hits_energy_conserved(hits, th, on_corrected):
    hits_thresh = threshold_hits(hits, th, on_corrected=on_corrected)
    assert_almost_equal(hits.E .sum(), hits_thresh.E .sum())
    assert_almost_equal(hits.Ec.sum(), hits_thresh.Ec.sum())


@mark.parametrize( "on_corrected  col".split()
                 , ( (     False, "Q" )
                   , (      True, "Qc")))
@given(hits=list_of_hits(), th=floats())
def test_threshold_hits_all_larger_than_th(hits, th, col, on_corrected):
    hits_thresh = threshold_hits(hits, th, on_corrected = on_corrected)
    non_nn = hits_thresh.loc[hits_thresh.Q != NN]
    q = non_nn[col]
    assert np.all(q >= th)
