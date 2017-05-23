import numpy as np
import scipy as sc

from . params import Measurement



class Correction:
    """
    Interface for accessing any kind of corrections.

    Parameters
    ----------
    xs : np.ndarray
        Array of coordinates corresponding to each correction.
    fs : np.ndarray
        Array of corrections or the values used for computing them.
    us : np.ndarray
        Array of uncertainties or the values used for computing them.
    norm_strategy : False or string
        Flag to set the normalization option. Accepted values:
        - False:    Do not normalize.
        - "max":    Normalize to maximum energy encountered.
        - "index":  Normalize to the energy placed to index (i,j).
    interp_strategy : string
        Flag to set the interpolation option. Accepted values:
        - "nearest"  : Take correction from the closest node
        - "bivariate": Cubic spline interpolation in 2d.
    default_f, default_u : floats
        Default correction and uncertainty for missing values (where fs = 0).
    """

    def __init__(self,
                 xs, fs, us,
                   norm_strategy = False, index = None,
                 interp_strategy = "nearest",
                 default_f = 0, default_u = 0):

        self._xs = [np.array( x, dtype=float) for x in xs]
        self._fs =  np.array(fs, dtype=float)
        self._us =  np.array(us, dtype=float)

        self._interp_strategy = interp_strategy

        self._default_f = default_f
        self._default_u = default_u

        self._normalize(norm_strategy, index)
        self._get_correction = self._define_interpolation(interp_strategy)

    def __call__(self, *x):
        """
        Compute the correction factor.

        Parameters
        ----------
        *x: Sequence of nd.arrays
             Each array is one coordinate. The number of coordinates must match
             that of the `xs` array in the init method.
        """
        return Measurement(*self._get_correction(*x))

    def _define_interpolation(self, opt):
        if   opt == "nearest"  : corr = self._nearest_neighbor
        elif opt == "bivariate": corr = self._bivariate()
        else: raise ValueError("Interpolation option not recognized: {}".format(opt))
        return corr

    def _normalize(self, strategy, index):
        if not strategy           : return
        elif   strategy == "max"  : index = np.argmax(self._fs)
        elif   strategy == "index": pass#index = index
        else: raise ValueError("Normalization option not recognized: {}".format(strategy))

        f_ref = self._fs[index]
        u_ref = self._us[index]

        valid_fs = self._fs > 0
        input_fs = self._fs.copy()

        # Redefine and propagate uncertainties as:
        # u(F) = F sqrt(u(F)**2/F**2 + u(Fref)**2/Fref**2)
        self._fs = f_ref / self._fs
        self._us = self._fs * np.sqrt((self._us / input_fs)**2 +
                                      (   u_ref / f_ref   )**2 )

        # Set invalid to defaults
        self._fs[~valid_fs] = self._default_f
        self._us[~valid_fs] = self._default_u

    def _find_closest_indices(self, x, y):
        # Find the index of the closest value in y for each value in x.
        return np.argmin(abs(x-y[:, np.newaxis]), axis=0)

    def _nearest_neighbor(self, *x):
        # Find the index of the closest value for each axis
        x_closest = tuple(map(self._find_closest_indices, x, self._xs))
        return self._fs[x_closest], self._us[x_closest]

    def _bivariate(self):
        f_interp = sc.interpolate.RectBivariateSpline(*self._xs, self._fs)
        u_interp = sc.interpolate.RectBivariateSpline(*self._xs, self._us)
        return lambda x, y: (f_interp(x, y), u_interp(x, y))


class Fcorrection:
    def __init__(self, f, u_f, pars):
        self._f   = lambda x:   f(x, *pars)
        self._u_f = lambda x: u_f(x, *pars)

    def __call__(self, x):
        return Measurement(self._f(x), self._u_f(x))
