from enum import Enum

import numpy        as np
import scipy.signal as signal
import scipy.stats  as stats

from functools import wraps

from .. core.core_functions import to_col_vector


class BlsMode(Enum):
    mean      = 0
    median    = 1
    scipymode = 2
    mode      = 3


def mask_sensors(wfs, active):
    return wfs * active.astype(wfs.dtype).reshape(active.size, 1)


def scipy_mode(x, axis=0):
    """
    Scipy implementation of the mode (runs very slow).
    Returns a column vector.
    """
    m, c = stats.mode(x, axis=axis)
    return m


def mode(wfs, axis=0):
    """
    A fast calculation of the mode: it runs 10 times
    faster than the SciPy version but only applies to
    positive waveforms.
    """
    def wf_mode(wf):
        positive = wf > 0
        return np.bincount(wf[positive]).argmax() if np.count_nonzero(positive) else 0
    return np.apply_along_axis(wf_mode, axis, wfs).astype(float)


def zero_masked(fn):
    """
    protection for mean and median
    so that we get the correct answer in case of 
    zero suppressed data
    """
    @wraps(fn)
    def proxy(wfs, *args, **kwds):
        mask_wfs = np.ma.masked_where(wfs == 0, wfs)
        return fn(mask_wfs, *args, **kwds).filled(0)
    proxy.__doc__ = "Masked version to protect ZS mode \n\n" + proxy.__doc__
    return proxy

median = zero_masked(np.ma.median)
mean   = zero_masked(np.ma.mean)


def means  (wfs): return to_col_vector(mean  (wfs, axis=1))
def medians(wfs): return to_col_vector(median(wfs, axis=1))
def modes  (wfs): return to_col_vector(mode  (wfs, axis=1))


def subtract_baseline(wfs, *, bls_mode=BlsMode.mean):
    """
    Subtract the baseline to all waveforms in the input
    with a specific algorithm.

    Parameters
    ----------
    wfs: np.ndarray with shape (n, m)
        Waveforms with baseline.

    Keyword-only parameters
    -----------------------
    bls_mode: BlsMode
        Algorithm to be used.

    Returns
    -------
    bls: np.ndarray with shape (n, m)
        Baseline-subtracted waveforms.
    """

    if   bls_mode is BlsMode.mean     : return wfs - means     (wfs)
    elif bls_mode is BlsMode.median   : return wfs - medians   (wfs)
    elif bls_mode is BlsMode.mode     : return wfs - modes     (wfs)
    elif bls_mode is BlsMode.scipymode: return wfs - scipy_mode(wfs, axis=1)
    else:
        raise TypeError(f"Unrecognized baseline subtraction option: {bls_mode}")
    return bls


def calibrate_wfs(wfs, adc_to_pes):
    """
    Convert waveforms in adc to pes. Masked channels
    are ignored.
    """
    adc_to_pes = to_col_vector(adc_to_pes)
    ok         = adc_to_pes > 0
    out        = np.zeros(wfs.shape, dtype=float)
    return np.divide(wfs, adc_to_pes, out=out, where=ok)


def subtract_baseline_and_calibrate(sipm_wfs, adc_to_pes, *, bls_mode=BlsMode.mean):
    bls = subtract_baseline(sipm_wfs, bls_mode=bls_mode)
    return calibrate_wfs(bls, adc_to_pes)


def calibrate_pmts(cwfs, adc_to_pes, n_MAU=100, thr_MAU=3):
    """
    This function is called for PMT waveforms that have
    already been baseline restored and pedestal subtracted.
    It computes the calibrated waveforms and its sensor sum.
    It also computes the calibrated waveforms and sensor
    sum for elements of the waveforms above some value
    (thr_MAU) over a MAU that follows the waveform. These
    are useful to suppress oscillatory noise and thus can
    be applied for S1 searches (the calibrated version
    without the MAU should be applied for S2 searches).
    """
    MAU         = np.full(n_MAU, 1 / n_MAU)
    mau         = signal.lfilter(MAU, 1, cwfs, axis=1)

    # ccwfs stands for calibrated corrected waveforms
    ccwfs       = calibrate_wfs(cwfs, adc_to_pes)
    ccwfs_mau   = np.where(cwfs >= mau + thr_MAU, ccwfs, 0)

    cwf_sum     = np.sum(ccwfs    , axis=0)
    cwf_sum_mau = np.sum(ccwfs_mau, axis=0)
    return ccwfs, ccwfs_mau, cwf_sum, cwf_sum_mau


def pmt_subtract_mau(cwfs, n_MAU=100):
    """
    Subtract a MAU from the input waveforms.
    """
    MAU         = np.full(n_MAU, 1 / n_MAU)
    mau         = signal.lfilter(MAU, 1, cwfs, axis=1)

    return cwfs - mau


def calibrate_sipms(sipm_wfs, adc_to_pes, thr, *, bls_mode=BlsMode.mode):
    """
    Subtracts the baseline, calibrates waveforms to pes
    and suppresses values below `thr` (in pes).
    """
    thr  = to_col_vector(np.full(sipm_wfs.shape[0], thr))
    bls  = subtract_baseline(sipm_wfs, bls_mode=bls_mode)
    cwfs = calibrate_wfs(bls, adc_to_pes)
    return np.where(cwfs > thr, cwfs, 0)


def subtract_mean  (wfs): return subtract_baseline(wfs, bls_mode=BlsMode.mean  )
def subtract_median(wfs): return subtract_baseline(wfs, bls_mode=BlsMode.median)
def subtract_mode  (wfs): return subtract_baseline(wfs, bls_mode=BlsMode.mode  )


def sipm_subtract_mode_and_calibrate  (sipm_wfs, adc_to_pes): return calibrate_wfs(subtract_mode  (sipm_wfs), adc_to_pes)
def sipm_subtract_mean_and_calibrate  (sipm_wfs, adc_to_pes): return calibrate_wfs(subtract_mean  (sipm_wfs), adc_to_pes)
def sipm_subtract_median_and_calibrate(sipm_wfs, adc_to_pes): return calibrate_wfs(subtract_median(sipm_wfs), adc_to_pes)


# Dict of functions for SiPM processing
sipm_processing = {
    'subtract_mode'            :      subtract_mode                ,# For gain extraction
    'subtract_median'          :      subtract_median              ,# For gain extraction
    'subtract_mode_calibrate'  : sipm_subtract_mode_and_calibrate  ,# For PDF calculation
    'subtract_mean_calibrate'  : sipm_subtract_mean_and_calibrate  ,# For PDF calculation
    'subtract_median_calibrate': sipm_subtract_median_and_calibrate,# For PDF calculation
    'subtract_mode_zs'         : calibrate_sipms                    # For data processing
}
