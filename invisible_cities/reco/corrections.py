from functools import partial
from itertools import product

import numpy as np
from scipy.interpolate import griddata

from ..core               import fit_functions as fitf
from ..core.exceptions    import ParameterNotSet
from .. evm.ic_containers import Measurement


opt_nearest = {"interp_method": "nearest"}
opt_linear  = {"interp_method": "linear" ,
               "default_f"    :     1    ,
               "default_u"    :     0    }
opt_cubic   = {"interp_method":  "cubic" ,
               "default_f"    :     1    ,
               "default_u"    :     0    }


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
    default_f, default_u : floats
        Default correction and uncertainty for missing values (where fs = 0).
    """

    def __init__(self,
                 xs, fs, us,
                   norm_strategy = None,
                   norm_opts     = {},
                 interp_method   = "nearest",
                 default_f       = 0,
                 default_u       = 0):

        self._xs = [np.array( x, dtype=float) for x in xs]
        self._fs =  np.array(fs, dtype=float)
        self._us =  np.array(us, dtype=float)

        self.norm_strategy   =   norm_strategy
        self.norm_opts       =   norm_opts
        self.interp_method   = interp_method
        self.default_f       = default_f
        self.default_u       = default_u

        self._normalize        (  norm_strategy,
                                  norm_opts    )
        self._init_interpolator(interp_method  , default_f, default_u)

    def __call__(self, *xs):
        """
        Compute the correction factor.

        Parameters
        ----------
        *x: Sequence of nd.arrays
             Each array is one coordinate. The number of coordinates must match
             that of the `xs` array in the init method.
        """
        # In order for this to work well both for arrays and scalars
        arrays = len(np.shape(xs)) > 1
        if arrays:
            xs = np.stack(xs, axis=1)

        value  = self._get_value      (xs).flatten()
        uncert = self._get_uncertainty(xs).flatten()
        return (Measurement(value   , uncert   ) if arrays else
                Measurement(value[0], uncert[0]))

    def _init_interpolator(self, method, default_f, default_u):
        coordinates           = np.array(list(product(*self._xs)))
        self._get_value       = partial(griddata,
                                        coordinates,
                                        self._fs.flatten(),
                                        method     = method,
                                        fill_value = default_f)

        self._get_uncertainty = partial(griddata,
                                        coordinates,
                                        self._us.flatten(),
                                        method     = method,
                                        fill_value = default_u)

    def _normalize(self, strategy, opts):
        if not strategy            : return

        elif   strategy == "const" :
            if "value" not in opts:
                raise ParameterNotSet(("Normalization strategy 'const' requires"
                                       "the normalization option 'value'"))
            f_ref = opts["value"]
            u_ref = 0

        elif   strategy == "max"   :
            flat_index = np.argmax(self._fs)
            mult_index = np.unravel_index(flat_index, self._fs.shape)
            f_ref = self._fs[mult_index]
            u_ref = self._us[mult_index]

        elif   strategy == "center":
            index = tuple(i // 2 for i in self._fs.shape)
            f_ref = self._fs[index]
            u_ref = self._us[index]

        elif   strategy == "index" :
            if "index" not in opts:
                raise ParameterNotSet(("Normalization strategy 'index' requires"
                                       "the normalization option 'index'"))
            index = opts["index"]
            f_ref = self._fs[index]
            u_ref = self._us[index]

        else:
            raise ValueError("Normalization strategy not recognized: {}".format(strategy))

        assert f_ref > 0, "Invalid reference value."

        valid    = (self._fs > 0) & (self._us > 0)
        valid_fs = self._fs[valid].copy()
        valid_us = self._us[valid].copy()

        # Redefine and propagate uncertainties as:
        # u(F) = F sqrt(u(F)**2/F**2 + u(Fref)**2/Fref**2)
        self._fs[ valid]  = f_ref / valid_fs
        self._us[ valid]  = np.sqrt((valid_us / valid_fs)**2 +
                                    (   u_ref / f_ref   )**2 )
        self._us[ valid] *= self._fs[valid]

        # Set invalid to defaults
        self._fs[~valid]  = self.default_f
        self._us[~valid]  = self.default_u

    def __eq__(self, other):
        for i, x in enumerate(self._xs):
            if not np.allclose(x, other._xs[i]):
                return False

        if not np.allclose(self._fs, other._fs):
            return False

        if not np.allclose(self._us, other._us):
            return False

        return True


class Fcorrection:
    def __init__(self, f, u_f, pars):
        self._f   = lambda *x:   f(*x, *pars)
        self._u_f = lambda *x: u_f(*x, *pars)

    def __call__(self, *x):
        return Measurement(self._f(*x), self._u_f(*x))


def LifetimeCorrection(LT, u_LT):
    fun   = lambda z, LT, u_LT=0: fitf.expo(z, 1, LT)
    u_fun = lambda z, LT, u_LT  : z * u_LT / LT**2 * fun(z, LT)
    return Fcorrection(fun, u_fun, (LT, u_LT))


def LifetimeXYCorrection(pars, u_pars, xs, ys, **kwargs):
    LTs = Correction((xs, ys), pars, u_pars, **kwargs)
    return (lambda z, x, y: LifetimeCorrection(*LTs(x, y))(z))


def LifetimeRCorrection(pars, u_pars):
    def LTfun(z, r, a, b, c, u_a, u_b, u_c):
        LT = a - b * r * np.exp(r / c)
        return fitf.expo(z, 1, LT)

    def u_LTfun(z, r, a, b, c, u_a, u_b, u_c):
        LT   = a - b * r * np.exp(r / c)
        u_LT = (u_a**2 + u_b**2 * np.exp(2 * r / c) +
                u_c**2 *   b**2 * r**2 * np.exp(2 * r / c) / c**4)**0.5
        return z * u_LT / LT**2 * LTfun(z, r, a, b, c, u_a, u_b, u_c)

    return Fcorrection(LTfun, u_LTfun, np.concatenate([pars, u_pars]))
