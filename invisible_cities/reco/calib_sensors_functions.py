from enum import Enum

import numpy        as np
import scipy.signal as signal
    
class BlsMode(Enum):
    mean = 0
    mau  = 1


def mode(wfs):
    gen_m = lambda x: np.bincount(x[x>=0]).argmax() if x[x>=0].size != 0 else -1
    return np.apply_along_axis(gen_m, 1, wfs)


def subtract_baseline(wfs, *, bls_mode=BlsMode.mean, **bls_opts):
    bls = wfs - np.mean(wfs, axis=1)[:, np.newaxis]
    if   bls_mode is BlsMode.mean:
        pass
    elif bls_mode is BlsMode.mau:
        n_MAU = bls_opts["n_MAU"]
        MAU   = np.full(n_MAU, 1 / n_MAU)
        mau   = signal.lfilter(MAU, 1, bls, axis=1)
        bls   = bls - mau
    else:
        raise TypeError(f"Unrecognized baseline subtraction option: {bls_mode}")
    return bls


def calibrate_wfs(wfs, adc_to_pes):
    adc_to_pes = adc_to_pes.reshape(adc_to_pes.size, 1)
    ok         = adc_to_pes > 0
    return np.divide(wfs, adc_to_pes, out=np.zeros_like(wfs), where=ok)


def calibrate_pmts(cwfs, adc_to_pes,
                   n_MAU = 100, thr_MAU = 3):
    MAU         = np.full(n_MAU, 1 / n_MAU)
    mau         = signal.lfilter(MAU, 1, cwfs, axis=1)

    # ccwfs stands for calibrated corrected waveforms
    ccwfs       = calibrate_wfs(cwfs, adc_to_pes)
    ccwfs_mau   = np.where(cwfs >= mau + thr_MAU, ccwfs, 0)

    cwf_sum     = np.sum(ccwfs    , axis=0)
    cwf_sum_mau = np.sum(ccwfs_mau, axis=0)
    return ccwfs, ccwfs_mau, cwf_sum, cwf_sum_mau


def pmt_subtract_mau(cwfs, n_MAU = 100):
    """ Returns the difference to the MAU """
    MAU = np.full(n_MAU, 1 / n_MAU)
    mau = signal.lfilter(MAU, 1, cwfs, axis=1)

    return cwfs - mau


def calibrate_sipms(sipm_wfs, adc_to_pes, thr, n_MAU=100):
    """
    Subtracts the baseline as an average of the waveform and
    uses a MAU to set the signal threshold (thr, in PES).
    """
    thr  = np.full(sipm_wfs.shape[0], thr)[:, np.newaxis]
    bls  = subtract_baseline(sipm_wfs)
    cwfs = calibrate_wfs(bls, adc_to_pes)
    MAU  = np.full(n_MAU, 1 / n_MAU)
    mau  = signal.lfilter(MAU, 1, cwfs, axis=1)
    return np.where(cwfs > mau + thr, cwfs, 0)


def subtract_mode(wfs):
    return wfs - mode(wfs)[:, np.newaxis]


def subtract_mean(wfs):
    return wfs - np.mean(wfs, axis=1)[:, np.newaxis]


def sipm_subtract_mode_and_calibrate(sipm_wfs, adc_to_pes):
    """Computes mode pedestal"""
    bls = subtract_mode(sipm_wfs)
    return calibrate_wfs(bls.astype(np.float), adc_to_pes)


def sipm_subtract_mean_and_calibrate(sipm_wfs, adc_to_pes):
    bls = subtract_mean(sipm_wfs)
    return calibrate_wfs(bls, adc_to_pes)


## Not 100% sure if we want to keep this?!?!?
def sipm_subtract_mean_mau_and_calibrate(sipm_wfs, adc_to_pes, n_MAU=100):
    bls = subtract_baseline(sipm_wfs, bls_mode=BlsMode.mau, n_MAU=n_MAU)
    return calibrate_wfs(bls, adc_to_pes)


def sipm_subtract_mode_select_signal(sipm_wfs, adc_to_pes, thr):
    zwfs = sipm_subtract_mode_and_calibrate(sipm_wfs, adc_to_pes)
    return np.where(zwfs > thr, zwf, 0)


def subtract_baseline_and_calibrate(sipm_wfs, adc_to_pes):
    """Computes pedetal as average of the waveform. Very fast"""
    bls = subtract_baseline(sipm_wfs)
    return calibrate_wfs(bls, adc_to_pes)


def subtract_baseline_mau_and_calibrate(sipm_wfs, adc_to_pes, n_MAU=100):
    """Computes pedetal using a MAU. Runs a factor 100 slower than previous"""
    bls = subtract_baseline(sipm_wfs, bls_mode=BlsMode.mau, n_MAU=n_MAU)
    return calibrate_wfs(bls, adc_to_pes)


## Dict of functions for SiPM processing
sipm_processing = {
    'gain'  : subtract_mode,
    'gainM' : subtract_mean,
    'pdf'   : sipm_subtract_mode_and_calibrate,
    'pdfM'  : sipm_subtract_mean_and_calibrate,
    'pdfMM' : sipm_subtract_mean_mau_and_calibrate,
    'data'  : sipm_subtract_mode_select_signal,
    'dataM' : calibrate_sipms
    }
