import os

from collections import namedtuple
from itertools   import starmap

import numpy  as np
import tables as tb

from pytest import fixture
from pytest import approx
from pytest import mark
from flaky  import flaky

from . import blr


deconv_params = namedtuple("deconv_params",
                           "n_baseline coeff_clean coeff_blr "
                           "thr_trigger accum_discharge_length")


@fixture(scope="session")
def sin_wf_params():
    n_baseline             = 500
    coeff_clean            = 1e-6
    coeff_blr              = 1e-3
    thr_trigger            = 1e-3
    accum_discharge_length = 10
    return deconv_params(n_baseline, coeff_clean, coeff_blr,
                         thr_trigger, accum_discharge_length)


@fixture
def sin_wf(sin_wf_params):
    baseline        = np.random.uniform(0, 100)
    wf              = np.full(sin_wf_params.n_baseline, baseline)
    start           = np.random.choice (sin_wf_params.n_baseline // 2)
    length          = np.random.randint(sin_wf_params.n_baseline // 50,
                                        sin_wf_params.n_baseline // 2)
    stop            = start + length

    # minus sign to simulate inverted pmt response
    wf[start:stop] -= np.sin(np.linspace(0, 2 * np.pi, length))
    return wf


@fixture(scope="session")
def ad_hoc_blr_signals(example_blr_wfs_filename):
    with tb.open_file(example_blr_wfs_filename) as file:
        rwf    = file.root.RWF[:]
        blrwf  = file.root.BLR[:]
        attrs  = file.root.BLR.attrs
        params = deconv_params(attrs.n_baseline            ,
                               attrs.coeff_c               ,
                               attrs.coeff_blr             ,
                               attrs.thr_trigger           ,
                               attrs.accum_discharge_length)
        return rwf, blrwf, params


def test_deconvolve_signal_positive_integral(sin_wf, sin_wf_params):
    # The RWF should have null integral because contains roughly
    # the same number of positive and negative samples.
    # The CWF on the other hand should contain mostly positive
    # samples, therefore the integral should be positive.
    blr_wf = blr.deconvolve_signal(sin_wf, **sin_wf_params._asdict())
    assert np.sum(blr_wf > 0)


def test_deconvolve_signal_baseline_is_recovered(sin_wf, sin_wf_params):
    # The RWF contains a baseline. The deconvolution process should
    # remove it. We take the baseline as the most repeated value.
    blr_wf      = blr.deconvolve_signal(sin_wf, **sin_wf_params._asdict())
    (entries,
     amplitude) = np.histogram(blr_wf, 200)
    baseline    = amplitude[np.argmax(entries)]
    assert np.abs(baseline) < 0.1


@mark.slow
@flaky(max_runs=3, min_passes=3)
def test_deconvolve_signal_ad_hoc_signals(ad_hoc_blr_signals):
    all_rwfs, all_true_blr_wfs, params = ad_hoc_blr_signals

    # This test takes long, so we pick just one waveform.
    # Its exhaustiveness relies on repeated test runs.
    nevt, npmt, _ = all_rwfs.shape
    evt_no = np.random.choice(nevt)
    pmt_no = np.random.choice(npmt)

    rwf         = all_rwfs        [evt_no, pmt_no]
    true_blr_wf = all_true_blr_wfs[evt_no, pmt_no]
    blr_wf = blr.deconvolve_signal(rwf.astype(np.double),
                                   n_baseline             = params.n_baseline,
                                   coeff_clean            = params.coeff_clean[pmt_no],
                                   coeff_blr              = params.coeff_blr  [pmt_no],
                                   thr_trigger            = params.thr_trigger,
                                   accum_discharge_length = params.accum_discharge_length)
    assert np.allclose(blr_wf, true_blr_wf)


@mark.slow
def test_deconv_pmt_ad_hoc_signals_all(ad_hoc_blr_signals):
    all_rwfs, all_true_blr_wfs, params = ad_hoc_blr_signals
    pmt_active                         = [] # Means all

    # This test takes long, so we pick a random event.
    # Its exhaustiveness relies on repeated test runs.
    evt_no           = np.random.choice(all_rwfs.shape[0])
    evt_rwfs         = all_rwfs        [evt_no]
    evt_true_blr_wfs = all_true_blr_wfs[evt_no]

    blr_wfs = blr.deconv_pmt(evt_rwfs,
                             params.coeff_clean,
                             params.coeff_blr  ,
                             pmt_active,
                             params.n_baseline ,
                             params.thr_trigger,
                             params.accum_discharge_length)

    np.allclose(blr_wfs, evt_true_blr_wfs)


@mark.slow
def test_deconv_pmt_ad_hoc_signals_dead_sensors(ad_hoc_blr_signals):
    all_rwfs, all_true_blr_wfs, params = ad_hoc_blr_signals

    n_evts, n_pmts, _ = all_rwfs.shape
    pmt_active        = np.arange(n_pmts)
    n_alive           = np.random.randint(1, n_pmts - 1)
    pmt_active        = np.random.choice(pmt_active, size=n_alive, replace=False)

    # This test takes long, so we pick a random event.
    # Its exhaustiveness relies on repeated test runs.
    evt_no           = np.random.choice(n_evts)
    evt_rwfs         = all_rwfs        [evt_no]
    evt_true_blr_wfs = all_true_blr_wfs[evt_no]

    blr_wfs = blr.deconv_pmt(evt_rwfs,
                             params.coeff_clean,
                             params.coeff_blr  ,
                             pmt_active.tolist(),
                             params.n_baseline ,
                             params.thr_trigger,
                             params.accum_discharge_length)

    np.allclose(blr_wfs, evt_true_blr_wfs[pmt_active])
