from collections import namedtuple

import numpy  as np
import tables as tb

from pytest import fixture
from pytest import mark
from flaky  import flaky

from .. reco import calib_sensors_functions as csf
from .       import blr


deconv_params = namedtuple("deconv_params",
                           "coeff_clean coeff_blr "
                           "thr_trigger accum_discharge_length")


@fixture(scope="session")
def sin_wf_params():
    n_baseline             = 500
    coeff_clean            = 1e-6
    coeff_blr              = 1e-3
    thr_trigger            = 1e-3
    accum_discharge_length = 10
    return n_baseline, deconv_params(coeff_clean, coeff_blr,
                                     thr_trigger, accum_discharge_length)


@fixture
def sin_wf(sin_wf_params):
    n_baseline, _   = sin_wf_params
    baseline        = np.random.uniform(0, 100)
    wf              = np.full(n_baseline, baseline)
    start           = np.random.choice (n_baseline // 2)
    length          = np.random.randint(n_baseline // 50, n_baseline // 2)
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
        params = deconv_params(attrs.coeff_c               ,
                               attrs.coeff_blr             ,
                               attrs.thr_trigger           ,
                               attrs.accum_discharge_length)
        return rwf, blrwf, attrs.n_baseline, params


def test_deconvolve_signal_positive_integral(sin_wf, sin_wf_params):
    # The RWF should have null integral because contains roughly
    # the same number of positive and negative samples.
    # The CWF on the other hand should contain mostly positive
    # samples, therefore the integral should be positive.
    n_baseline, params = sin_wf_params
    sin_wf_noped = np.mean(sin_wf[:n_baseline]) - sin_wf
    blr_wf = blr.deconvolve_signal(sin_wf_noped, **params._asdict())
    assert np.sum(blr_wf > 0)


def test_deconvolve_signal_baseline_is_recovered(sin_wf, sin_wf_params):
    # The RWF contains a baseline. The deconvolution process should
    # remove it. We take the baseline as the most repeated value.
    n_baseline, params = sin_wf_params
    sin_wf_noped = np.mean(sin_wf[:n_baseline]) - sin_wf
    blr_wf      = blr.deconvolve_signal(sin_wf_noped, **params._asdict())
    (entries,
     amplitude) = np.histogram(blr_wf, 200)
    baseline    = amplitude[np.argmax(entries)]
    assert np.abs(baseline) < 0.1


@mark.slow
@flaky(max_runs=3, min_passes=3)
def test_deconvolve_signal_ad_hoc_signals(ad_hoc_blr_signals):
    all_rwfs, all_true_blr_wfs, n_baseline, params = ad_hoc_blr_signals

    # This test takes long, so we pick just one waveform.
    # Its exhaustiveness relies on repeated test runs.
    nevt, npmt, _ = all_rwfs.shape
    evt_no = np.random.choice(nevt)
    pmt_no = np.random.choice(npmt)

    rwf         = all_rwfs        [evt_no, pmt_no]
    true_blr_wf = all_true_blr_wfs[evt_no, pmt_no]
    
    cwf         = np.mean(rwf[:n_baseline]) - rwf
    blr_wf = blr.deconvolve_signal(cwf,
                                   coeff_clean            = params.coeff_clean[pmt_no],
                                   coeff_blr              = params.coeff_blr  [pmt_no],
                                   thr_trigger            = params.thr_trigger,
                                   accum_discharge_length = params.accum_discharge_length)
    assert np.allclose(blr_wf, true_blr_wf)


@mark.slow
def test_deconv_pmt_ad_hoc_signals_all(ad_hoc_blr_signals):
    all_rwfs, all_true_blr_wfs, n_baseline, params = ad_hoc_blr_signals

    # This test takes long, so we pick a random event.
    # Its exhaustiveness relies on repeated test runs.
    evt_no           = np.random.choice(all_rwfs.shape[0])
    evt_rwfs         = all_rwfs        [evt_no]
    evt_true_blr_wfs = all_true_blr_wfs[evt_no]

    evt_cwfs = csf.means(evt_rwfs[:, :n_baseline]) - evt_rwfs

    rep_thr  = np.repeat(params.thr_trigger           , evt_cwfs.shape[0])
    rep_acc  = np.repeat(params.accum_discharge_length, evt_cwfs.shape[0])
    blr_wfs  = np.array(tuple(map(blr.deconvolve_signal, evt_cwfs        ,
                                  params.coeff_clean   , params.coeff_blr,
                                  rep_thr              , rep_acc         )))

    np.allclose(blr_wfs, evt_true_blr_wfs)


@mark.slow
def test_deconv_pmt_ad_hoc_signals_dead_sensors(ad_hoc_blr_signals):
    all_rwfs, all_true_blr_wfs, n_baseline, params = ad_hoc_blr_signals

    n_evts, n_pmts, _ = all_rwfs.shape
    pmt_active        = np.arange(n_pmts)
    n_alive           = np.random.randint(1, n_pmts - 1)
    pmt_active        = np.random.choice(pmt_active, size=n_alive, replace=False)

    # This test takes long, so we pick a random event.
    # Its exhaustiveness relies on repeated test runs.
    evt_no           = np.random.choice(n_evts)
    evt_rwfs         = all_rwfs        [evt_no]
    evt_true_blr_wfs = all_true_blr_wfs[evt_no]

    evt_cwfs = csf.means(evt_rwfs[:, :n_baseline]) - evt_rwfs
    rep_thr  = np.repeat(params.thr_trigger           , len(pmt_active))
    rep_acc  = np.repeat(params.accum_discharge_length, len(pmt_active))
    blr_wfs  = np.array(tuple(map(blr.deconvolve_signal, evt_cwfs[pmt_active],
                                  params.coeff_clean   , params.coeff_blr    ,
                                  rep_thr              , rep_acc             )))

    np.allclose(blr_wfs, evt_true_blr_wfs[pmt_active])
