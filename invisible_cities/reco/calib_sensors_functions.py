from enum import Enum

import numpy        as np
import scipy.signal as signal


class BlsMode(Enum):
    mean = 0
    mau  = 1


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


def sipm_subtract_baseline_and_calibrate(sipm_wfs, adc_to_pes):
    """Computes pedetal as average of the waveform. Very fast"""
    bls = subtract_baseline(sipm_wfs)
    return calibrate_wfs(bls, adc_to_pes)


def sipm_subtract_baseline_mau_and_calibrate(sipm_wfs, adc_to_pes, n_MAU=100):
    """Computes pedetal using a MAU. Runs a factor 100 slower than previous"""
    bls = subtract_baseline(sipm_wfs, bls_mode=BlsMode.mau, n_MAU=n_MAU)
    return calibrate_wfs(bls, adc_to_pes)


def sipm_subtract_baseline_and_zs_mau(sipm_wfs, adc_to_pes, thr, n_MAU=100):
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
