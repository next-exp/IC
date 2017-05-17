import numpy as np
from pytest import fixture

from invisible_cities.reco.params            import Cluster
from invisible_cities.core.system_of_units_c import units
from invisible_cities.reco.xy_algorithms     import corona, barycenter, \
                                                select_sipms, discard_sipms, \
                                                get_nearby_sipm_inds

def test_barycenter_with_many_toy_signals():
    size=100
    for i in range(1000):
        xs = np.random.uniform(size=size)
        ys = np.random.uniform(size=size)
        qs = np.random.uniform(size=size)
        B  = barycenter(xs,ys,qs)
        assert np.isclose(B.X, (xs * qs).sum() / qs.sum())
        assert np.isclose(B.Y, (ys * qs).sum() / qs.sum())
        assert np.isclose(B.Q, qs.sum())
        assert B.Nsipm == size
    return

@fixture
def toy_sipm_signal():
    xs = np.array([65, -65]) * units.mm
    ys = np.array([65, -65]) * units.mm
    qs = np.ones (2        ) * units.pes * 5
    return xs, ys, qs

def test_corona_barycenter_are_same_with_one_cluster(toy_sipm_signal):
    xs, ys, qs = toy_sipm_signal
    c_clusters = corona(xs,ys,qs, rmax = 10 *units.m  , msipm = 1,
                                  Qlm  = 4.9*units.pes, Qthr  = 0)
    c_cluster = c_clusters[0]
    b_cluster = barycenter(xs,ys,qs)

    assert len(c_cluster)  == len(b_cluster)
    assert c_cluster.X     == b_cluster.X
    assert c_cluster.Y     == b_cluster.Y
    assert c_cluster.Q     == b_cluster.Q
    assert c_cluster.Nsipm == b_cluster.Nsipm
    assert c_cluster.Xrms  == b_cluster.Xrms
    assert c_cluster.Yrms  == b_cluster.Yrms

def test_corona_multiple_clusters(toy_sipm_signal):
    xs, ys, qs = toy_sipm_signal
    clusters = corona(xs, ys, qs, msipm=1, rmax=15*units.mm, Qlm=4.9*units.pes)
    assert len(clusters) == 2
    for i in range(len(xs)):
        assert clusters[i].X == xs[i]
        assert clusters[i].Y == ys[i]
        assert clusters[i].Q == qs[i]

def test_corona_min_threshold_Qthr():
    xs = np.arange(100) * 10
    ys = np.zeros (100)
    qs = np.arange(100)
    clusters = corona(xs, ys, qs,
                      Qthr  = 99*units.pes,
                      Qlm   = 1*units.pes,
                      slm   = 1*units.mm ,
                      rmax  = 2*units.mm ,
                      msipm = 1          )
    assert len(clusters) ==   1
    assert clusters[0].Q ==  99
    assert clusters[0].X == 990
    assert clusters[0].Y ==   0

def test_corona_msipm(toy_sipm_signal):
    xs, ys, qs = toy_sipm_signal
    assert len(corona(xs, ys, qs, msipm=2)) == 0

def test_corona_threshold_for_local_max(toy_sipm_signal):
    xs, ys, qs = toy_sipm_signal
    assert len(corona(xs, ys, qs,
                      msipm =  1,
                      Qlm     =  5.1*units.pes,
                      rmax  = 15  *units.mm )) == 0

def test_corona_rmax(toy_sipm_signal):
    xs, ys, qs = toy_sipm_signal
    assert len(corona(xs, ys, qs,
                      msipm =  1,
                      Qlm     =   4.9*units.pes,
                      rmax  =  15  *units.mm)) == 2

    assert len(corona(xs, ys, qs,
                      msipm = 1,
                      Qlm     = 4.9    *units.pes,
                      rmax  = 1000000*units.m)) == 1

@fixture
def toy_sipm_signal_and_inds():
    k = 10000
    xs = np.arange(k)
    ys = xs + k
    qs = xs + 2*k
    i  = np.array([0, 5, 1000, 9999])
    return k, i, xs, ys, qs

def test_select_sipms(toy_sipm_signal_and_inds):
    k, i, xs, ys, qs = toy_sipm_signal_and_inds
    xsel, ysel, qsel = select_sipms(i, xs, ys, qs)
    assert (xsel == i      ).all()
    assert (ysel == i +   k).all()
    assert (qsel == i + 2*k).all()

def test_discard_sipms(toy_sipm_signal_and_inds):
    k, i, xs, ys, qs = toy_sipm_signal_and_inds
    xsel, ysel, qsel = discard_sipms(i, xs, ys, qs)
    for ind in i:
        assert ind         not in xsel
        assert ind +     k not in ysel
        assert ind + 2 * k not in qsel

def test_get_nearby_sipm_inds():
    xs = np.array([0,1,2,3,4,0,1,2,3,4,0,1,2,3,4])
    ys = np.array([0,0,0,1,1,1,2,2,2,3,3,3,4,4,4])
    qs = np.ones  (100) * 10*units.pes
    xc, yc = (2, 2)
    d  = 1.5
    sis = get_nearby_sipm_inds(xc, yc, d, xs, ys, qs)
    for i in range(len(xs)):
        if i in sis:
            assert np.sqrt((xs[i] - xc)**2 + (ys[i] - yc)**2) <= d
        else:
            assert np.sqrt((xs[i] - xc)**2 + (ys[i] - yc)**2) >  d
