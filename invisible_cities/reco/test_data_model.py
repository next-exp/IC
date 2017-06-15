import numpy as np

from pytest import fixture
from pytest import mark
from pytest import raises
parametrize = mark.parametrize

from hypothesis             import given
from hypothesis.strategies  import floats
from hypothesis.strategies  import integers
from hypothesis.strategies  import composite
from hypothesis.extra.numpy import arrays

from .. core.system_of_units_c import units
from .. core.ic_types          import xy
from .. core.exceptions        import SipmEmptyList
from .. core.exceptions        import SipmZeroCharge

from .       event_model       import Cluster
from .       event_model       import Hit



@composite
def cluster_input(draw, min_value=0, max_value=100):
    x   = draw(floats(1.,5.))
    y   = draw(floats(-10.,10.))
    xrms   = draw(floats(0.01,0.5))
    yrms   = draw(floats(0.1,0.9))
    Q   = draw(floats(50.,100.))
    nsipm   = draw(integers(1,20))
    return Q, x, y, xrms, yrms, nsipm

@composite
def hit_input(draw, min_value=0, max_value=100):
    z   = draw(floats(0.1,0.9))
    s2_energy   = draw(floats(50.,100.))
    peak_number   = draw(integers(1,20))
    return peak_number, s2_energy, z


@given(cluster_input(1))
def test_cluster(ci):
    Q, x, y, xrms, yrms, nsipm = ci
    c = Cluster(Q, xy(x,y), xy(xrms,yrms), nsipm)
    assert c.nsipm      == nsipm

    np.isclose(c.Q, Q, rtol=1e-4)
    np.isclose(c.X, x, rtol=1e-4)
    np.isclose(c.Y, y, rtol=1e-4)
    np.isclose(xy.Y, b, rtol=1e-4)
    np.isclose(c.Xrms, xrms, rtol=01e-4)
    np.isclose(c.Yrms, yrms, rtol=1e-4)
    np.isclose(np.array(c.XY), np.array((x,y)), rtol=1e-4)
    np.isclose(c.R, np.sqrt(x ** 2 + y ** 2), rtol=1e-4)
    np.isclose(c.Phi, np.arctan2(y, x), rtol=1e-4)
    np.allclose(c.pos, np.stack(([x], [y]), axis=1), rtol=1e-4)

@given(cluster_input(1), hit_input(1))
def test_cluster(ci, hi):
    Q, x, y, xrms, yrms, nsipm = ci
    peak_number, s2_energy, z   = hi
    c = Cluster(Q, xy(x,y), xy(xrms,yrms), nsipm)
    h = Hit(peak_number, c, z, s2_energy)
    assert h.peak_number      == peak_number
    assert h.npeak            == peak_number
    np.isclose(h.z, z, rtol=1e-4)
    np.isclose(h.Z, z, rtol=1e-4)
    np.isclose(h.s2_energy, s2_energy, rtol=1e-4)
    np.isclose(h.E, s2_energy, rtol=1e-4)
    np.isclose(np.array(h.XYZ), np.array((x,y,z)), rtol=1e-4)
    np.allclose(h.VXYZ, np.array([x,y,z]), rtol=1e-4)
