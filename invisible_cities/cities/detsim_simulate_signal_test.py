import numpy  as np

from collections import namedtuple
from operator    import itemgetter

from pytest import fixture

import invisible_cities.core.system_of_units as units

from invisible_cities.core.core_functions  import in_range

from invisible_cities.cities.detsim_simulate_signal import pes_at_pmts
from invisible_cities.cities.detsim_simulate_signal import S1_TIMES
from invisible_cities.cities.detsim_simulate_signal import generate_S1_times_from_pes
# import create_xy_function
# import create_xyz_function

def create_xyz_function(H    : np.ndarray,
                        bins : list):
    """Given a 3D array and a list of bins for
    each dim, it returns a x,y,z function
    Parameters:
        :H: np.ndarray
            3D histogram
        :bins: list[np.ndarray, np.ndarray, np.ndarray]
            list with the bin edges of :H:. The i-element corresponds
            with the bin edges of the i-axis of :H:.
    Returns:
        :function:
            x, y, z function that returns the correspondig histogram value
    """

    xbins, ybins, zbins = bins
    if not H.shape == (len(xbins)-1, len(ybins)-1, len(zbins)-1):
        raise Exception("bins and array shapes not consistent")

    def function(x, y, z):
        if not x.shape==y.shape==z.shape:
            raise Exception("x, y and z must have same size")

        out = np.zeros(x.shape)
        #select values inside bin extremes
        selx = in_range(x, xbins[0], xbins[-1])
        sely = in_range(y, ybins[0], ybins[-1])
        selz = in_range(z, zbins[0], zbins[-1])
        sel = selx & sely & selz

        ix = np.digitize(x[sel], xbins)-1
        iy = np.digitize(y[sel], ybins)-1
        iz = np.digitize(z[sel], zbins)-1

        out[sel] = H[ix, iy, iz]
        return out
    return function

def create_xy_function(H    : np.ndarray,
                       bins : list):
    """Given a 2D array and a list of bins for
    each dim, it returns a x,y,z function
    Parameters:
        :H: np.ndarray
            2D histogram
        :bins: list[np.ndarray, np.ndarray]
            list with the bin edges of :H:. The i-element corresponds
            with the bin edges of the i-axis of :H:.
    Returns:
        :function:
            x, y function that returns the correspondig histogram value
    """

    xbins, ybins = bins
    if not H.shape == (len(xbins)-1, len(ybins)-1):
        raise Exception("bins and array shapes not consistent")

    def function(x, y):
        if not x.shape==y.shape:
            raise Exception("x, y and z must have same size")

        out = np.zeros(x.shape)
        #select values inside bin extremes
        selx = in_range(x, xbins[0], xbins[-1])
        sely = in_range(y, ybins[0], ybins[-1])
        sel = selx & sely

        ix = np.digitize(x[sel], xbins)-1
        iy = np.digitize(y[sel], ybins)-1

        out[sel] = H[ix, iy]
        return out
    return function


@fixture(scope="session")
def dummy_S2LT():
    """Cretes a dummy LT function:
        -Input x, y: vectors of same size
        -Output: array of size (vector size, number of pmts)"""

    xmin, xmax, dx = 0, 4, 2
    ymin, ymax, dy = 0, 4, 2

    xbins = np.arange(xmin, xmax+dx, dx)
    ybins = np.arange(ymin, ymax+dy, dy)
    bins = [xbins, ybins]

    # dummy light tables for 2 pmts
    Hpmt1 = np.array([[1, 1],
                      [1, 1]])

    Hpmt2 = np.array([[2, 2],
                      [2, 2]])
    Hs = [Hpmt1, Hpmt2]

    ###### CREATE XY FUNCTION FOR EACH SENSOR ######
    func_per_sensor = []
    for H in Hs:
        fxy = create_xy_function(H, bins)
        func_per_sensor.append(fxy)

    ###### CREATE XY CALLABLE FOR LIST OF XY FUNCTIONS #####
    def merge_list_of_functions(list_of_functions):
        def merged(x, y):
            return np.array([f(x, y) for f in list_of_functions]).T
        return merged

    return merge_list_of_functions(func_per_sensor)


def test_pes_at_pmts(dummy_S2LT):

    LT = dummy_S2LT

    x = np.array([1, 1 , 3])
    y = np.array([1, 10, 3])
    photons = np.array([10, 20, 30])

    pes = pes_at_pmts(LT, photons, x, y)

    #shape
    assert pes.shape == (2, 3)

    # is integer
    assert issubclass(pes.dtype.type, (np.integer, int))

    # out of bin edges
    assert np.all(pes[:,1] == np.zeros(2))


def test_generate_S1_times_from_pes():

    npmts = 10
    nhits = 2

    S1pes_pmt = np.array([[1, 1, 2, 2],
                          [1, 1, 1, 1]])
    S1times = generate_S1_times_from_pes(S1pes_pmt)

    for p, t in zip(S1pes_pmt, S1times):
        assert np.sum(p) == len(t)
