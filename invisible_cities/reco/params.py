import sys
from collections import namedtuple

import numpy as np
import scipy as sc


this_module = sys.modules[__name__]

def _add_namedtuple_in_this_module(name, attribute_names):
    new_nametuple = namedtuple(name, attribute_names)
    setattr(this_module, name, new_nametuple)

for name, attrs in (
        ('RawVectors'     , 'event pmtrwf sipmrwf pmt_active sipm_active'),
        ('SensorParams'   , 'NPMT PMTWL NSIPM SIPMWL'),
        ('CalibParams'    , 'coeff_c, coeff_blr, adc_to_pes_pmt adc_to_pes_sipm'),
        ('DeconvParams'   , 'n_baseline thr_trigger'),
        ('CalibVectors'   , 'channel_id coeff_blr coeff_c adc_to_pes adc_to_pes_sipm pmt_active'),
        ('S12Params'      , 'tmin tmax stride lmin lmax rebin'),
        ('PmapParams'     ,'s1_params s2_params s1p_params s1_PMT_params s1p_PMT_params'),
        ('ThresholdParams', 'thr_s1 thr_s2 thr_MAU thr_sipm thr_SIPM'),
        ('CalibratedSum'  , 'csum csum_mau'),
        ('CalibratedPMT'  , 'CPMT CPMT_mau'),
        ('S1PMaps'        , 'S1 S1_PMT S1p S1p_PMT'),
        ('PMaps'          , 'S1 S2 S2Si'),
        ('Peak'           , 't E'),
        ('FitFunction'    , 'fn values errors chi2 pvalue'),
        ('Cluster'        , 'Q pos rms Nsipm')):
    _add_namedtuple_in_this_module(name, attrs)

# Leave nothing but the namedtuple types in the namespace of this module
del name, namedtuple, sys, this_module, _add_namedtuple_in_this_module


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

        if   interp_strategy == "nearest"  : corr = self._nearest_neighbor
        elif interp_strategy == "bivariate": corr = self._bivariate()
        else: raise ValueError("Interpolation option not recognized: {}".format(interp_strategy))

        self.get_correction = corr


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
        return self.get_correction(*x)

