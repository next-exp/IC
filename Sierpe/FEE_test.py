# FEE_test.py
from math import sqrt
import pytest
from pytest import raises, mark
from hypothesis import given
from hypothesis.strategies import lists, floats
from flaky import flaky

import Sierpe.FEE as FE
import Core.system_of_units as units
import numpy as np

@mark.parametrize('a b'.split(),
                 ((1,1),(2,2),(4,4)))
def test_f(a,b):
    assert a==b

def test_sqrt():
    with raises (ValueError):
        sqrt(-1)

# def mean_list(lst):
#     return sum(lst)/len(lst)
#
# # @mark.parametrize('lst'.split(),
# #                  ((range(10),),([0,1,3,5],),([-3,-5],)))
# @given(lists(floats(allow_nan=False, allow_infinity=False), min_size=1))
# def test_mean(lst):
#     assert min(lst) <= mean_list(lst) <= max(lst)

def signal_i_th():
    """Generates a "theoretical" current signal (signal_i)"""
    return np.concatenate((np.zeros(1000),
                           np.linspace(0,0.5,2000),
                           np.linspace(0.5,0,1000),
                           np.linspace(0,1,1000),np.linspace(1,0,2000),
                           np.linspace(0,1,2000),np.linspace(1,0,2000),
                           np.zeros(20000)),axis=0)*units.mA

def deconv_simple(signal,coef):
    """Deconvolution of the fine-grained fee signal (no DAQ) no noise
    using true start and end of signals.
    """

    acum = np.zeros(len(signal))

    acum[0]=coef*signal[0]
    for n in np.arange(1,len(signal)):
        acum[n] = acum[n-1] + signal[n]

    signal_r = signal + coef*acum

    return signal_r, acum


def test_fee_params():
    """Check the values of the FEE params.

    These are parameters of the simulation. They can be changed but the test
    gurantess that changes must be propagated to test (e.g, changes
    are checked)
    """
    assert FE.PMT_GAIN == 1.7e6
    assert FE.FEE_GAIN == 582.237*units.ohm
    assert FE.DAQ_GAIN == 1.25
    assert FE.NBITS == 12
    assert FE.LSB == 2.0*units.V/2**FE.NBITS/FE.DAQ_GAIN
    assert FE.NOISE_I == FE.LSB/(FE.FEE_GAIN*FE.DAQ_GAIN)
    assert FE.NOISE_DAQ == 0.313*units.mV
    assert FE.C2 == 8*units.nF
    assert FE.C1 == 2714*units.nF
    assert FE.R1 == 1567*units.ohm
    assert FE.Zin == 62*units.ohm
    assert FE.t_sample == 25*units.ns
    assert FE.f_sample == 1./FE.t_sample
    assert FE.f_mc == 1./(1*units.ns)
    assert FE.f_LPF1 == 3*units.MHZ
    assert FE.f_LPF2 == 10*units.MHZ
    assert FE.OFFSET == 2500   # offset adc

def test_spe_to_adc():
    """Convert SPE to adc values with the current FEE Parameters must be."""
    ipmt = 0
    spe = FE.SPE()
    fee = FE.FEE(noise_FEEPMB_rms=0*units.mA, noise_DAQ_rms=0)

    spe_i = FE.spe_pulse(spe,t0=100*units.ns, tmax=200*units.ns)
    spe_v = FE.signal_v_fee(fee, spe_i,ipmt)
    spe_v_lpf = FE.signal_v_lpf(fee, spe_i)
    spe_adc = FE.daq_decimator(1000.*units.MHZ, 40*units.MHZ,
                               spe_v*FE.v_to_adc())
    spe_adc_lpf = FE.daq_decimator(1000.*units.MHZ, 40*units.MHZ,
                                   spe_v_lpf*FE.v_to_adc())
    adc_to_pes = np.sum(spe_adc[3:7])
    adc_to_pes_lpf = np.sum(spe_adc_lpf[3:7])
    assert adc_to_pes > 23
    assert adc_to_pes < 23.1
    assert adc_to_pes_lpf > 24
    assert adc_to_pes_lpf < 24.1

@flaky(max_runs=10, mi_passes=5)
@pytest.mark.feetest
def test_FEE():
    """
    1. starts from a "theoretical function " signal_i =signal_i_th()
    2. The effect of FEE is passed to signal_out (adding front-end noise)
    3. signal_outn adds the DAQ (adc) noise
    4. signal_out_cf is the cleaned signal (soft deconvolution)
    5. signal_r is the deconvoluted signal
    6. difference between input signal and deconvoluted signal
       must be small than 0.05 per thousand
    """
    signal_i = signal_i_th()

    fee = FE.FEE(noise_FEEPMB_rms=1*FE.NOISE_I, noise_DAQ_rms=FE.NOISE_DAQ)
    signal_out = FE.signal_v_fee(fee, signal_i, -1)
    signal_outn = FE.noise_adc(fee, signal_out * FE.v_to_adc())
    signal_out_cf = FE.signal_clean(fee, signal_outn, -1)
    signal_r, acum = deconv_simple(signal_out_cf,
                                   coef=fee.freq_LHPFd*np.pi)

    energy_mea = np.sum(signal_r[0:11000])
    energy_in = np.sum(signal_i[0:11000] * FE.i_to_adc())
    diff = 1000 * abs((energy_in - energy_mea)/energy_in)

    assert diff < 0.5
