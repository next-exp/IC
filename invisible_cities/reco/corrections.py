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
        - "center": Normalize to the energy placed at the center of the array.
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

        self.xs = [np.array( x, dtype=float) for x in xs]
        self.fs =  np.array(fs, dtype=float)
        self.us =  np.array(us, dtype=float)

        self.interp_strategy = interp_strategy

        self.default_f = default_f
        self.default_u = default_u

        self._normalize(norm_strategy, index)
        self.get_correction = self._define_interpolation(interp_strategy)


    def _define_interpolation(self, opt):
        if   opt == "nearest"  : corr = self._nearest_neighbor
        elif opt == "bivariate": corr = self._bivariate()
        else: raise ValueError("Interpolation option not recognized: {}".format(opt))
        return corr

    def _normalize(self, opt, index):
        if not opt           : return
        elif   opt == "max"  : index = np.argmax(self.fs)
        elif   opt == "index": index = tuple(index)
        else: raise ValueError("Normalization option not recognized: {}".format(opt))

        f_ref = self.fs[index]
        u_ref = self.us[index]

        valid_fs = self.fs > 0
        input_fs = self.fs.copy()

        # Redefine and propagate uncertainties as:
        # u(F) = F sqrt(u(F)**2/F**2 + u(Fref)**2/Fref**2)
        self.fs = f_ref / self.fs
        self.us = self.fs * np.sqrt((self.us/input_fs)**2 +
                                    (  u_ref/f_ref   )**2 )

        # Set invalid to defaults
        self.fs[~valid_fs] = self.default_f
        self.us[~valid_fs] = self.default_u

    def _find_closest_indices(self, x, y):
        # Find the index of the closest value in y for each value in x.
        return np.argmin(abs(x-y[:, np.newaxis]), axis=0)

    def _nearest_neighbor(self, *x):
        # Find the index of the closest value for each axis
        x_closest = list(map(self._find_closest_indices, x, self.xs))

        return self.fs[x_closest], self.us[x_closest]

    def _bivariate(self):
        f_interp = sc.interpolate.RectBivariateSpline(*self.xs, self.fs)
        u_interp = sc.interpolate.RectBivariateSpline(*self.xs, self.us)
        return lambda x, y: (f_interp(x, y), u_interp(x, y))

    def __call__(self, *x):
        """
        Compute the correction factor.
        
        Parameters
        ----------
        *x: Sequence of nd.arrays
             Each array is one coordinate. The number of coordinates must match
             that of the `xs` array in the init method.
        """
        return Measurement(*self.get_correction(*x))

