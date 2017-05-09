"""Classes and functions describing the electronics of the
PMT plane FEE.
(full model)
VH, JJGC, November, 2016
"""

import numpy as np
from scipy import signal
import invisible_cities.core.system_of_units as units
import invisible_cities.database.load_db as DB

# globals describing FEE
PMT_GAIN = 1.7e6
FEE_GAIN = 582.237 * units.ohm
DAQ_GAIN = 1.25
NBITS = 12
LSB = 2 * units.V / 2 ** NBITS / DAQ_GAIN
NOISE_I = LSB / (FEE_GAIN * DAQ_GAIN)
NOISE_DAQ = 0.313 * units.mV

C2 =    8 * units.nF
C1 = 2714 * units.nF
R1 = 1567 * units.ohm
Zin =  62 * units.ohm
t_sample = 25 * units.ns
f_sample = 1 / t_sample
f_mc = 1 / (1 * units.ns)
f_LPF1 =  3 * units.MHZ
f_LPF2 = 10 * units.MHZ
ADC_TO_PES_LPF = 24.1  # After LPF, comes out from spe area
ADC_TO_PES     = 23.1
OFFSET = 2500  # offset adc
CEILING = 4096  # ceiling of adc


def i_to_adc():
    """Current to adc counts."""
    return FEE_GAIN / LSB


def i_to_v():
    """Current to voltage."""
    return FEE_GAIN


def v_to_adc():
    """Voltage to adc."""
    return 1 / LSB


class SPE:
    """Represent a single photo-electron in the PMT."""

    def __init__(self, pmt_gain=PMT_GAIN, x_slope=5 * units.ns,
                 x_flat=1 * units.ns):

        self.pmt_gain = pmt_gain
        self.x_slope = x_slope
        self.x_flat = x_flat
        self.spe_base =       self.x_slope + self.x_flat
        self.spe_length = 2 * self.x_slope + self.x_flat

        # current
        self.A = self.pmt_gain*units.eplus / self.spe_base

        time_step = 1.*units.ns
        self.t = np.arange(0, self.spe_length, time_step)
        nns = int(self.x_slope / time_step)
        nnf = int(self.x_flat  / time_step)
        rise = np.linspace(0, self.A, num=nns)
        fall = np.linspace(self.A, 0, num=nns)
        flat = self.A*np.ones(nnf)
        self.spe = np.concatenate((rise, flat, fall))

    def __str__(self):
        s = """
        (PMT gain = {0:5.2g}, amplitude = {1:5.2g} muA
         slope = {2:5.2f} ns, flat = {3:5.2f} ns)
        """.format(self.pmt_gain, self.A / units.muA,
                   self.x_slope / units.ns,
                   self.x_flat  / units.ns)
        return s

    def __repr__(self):
        return self.__str__()


def spe_pulse(spe, t0=100 * units.ns, tmax=200 * units.ns,
              time_step=1 * units.ns):
    """
    input: an instance of class SPE
    Returns a SPE pulse at time t0
    with baseline extending in steps of time_step from 0 to tmax
    determined by DELTA_L
    """
    n = int(t0 / time_step)
    nmax = int(tmax / time_step)

    DELTA = np.zeros(nmax)   # Dirac delta of size DELTA_L
    DELTA[n] = 1
    # step = time_step/units.ns
    # spe_pulse_t =np.arange(0, len(DELTA) + len(self.spe) -1, step)
    spe_pulse = signal.convolve(DELTA, spe.spe)

    return spe_pulse


def spe_pulse_train(spe,
                    signal_start  = 2000 * units.ns,
                    signal_length = 5000 * units.ns,
                    daq_window    =   20 * units.mus,
                    time_step     =    1 * units.ns):
    """
    Input: an instance of class SPE
    Returns a train of SPE pulses between signal_start
    and start+length in daq_window separated by tstep
    """
    nmin = int(signal_start / time_step)
    nmax = int((signal_start + signal_length) / time_step)
    NMAX = int(daq_window / time_step)
    # step = time_step / units.ns

    DELTA = np.zeros(NMAX)
    DELTA[nmin:nmax+1] = 1
    # spe_pulse_t =np.arange(0,len(DELTA) + len(self.spe) -1,step)
    spe_pulse = signal.convolve(DELTA, spe.spe)

    return spe_pulse


def spe_pulse_from_vector(spe, cnt):
    """
    input: an instance of spe
    Returns a train of SPE pulses corresponding to vector cnt
    """

    spe_pulse = signal.convolve(cnt[0:-len(spe.spe)+1], spe.spe)
    return spe_pulse


class FEE:
    """Complete model of Front-end electronics."""

    def __init__(self, gain=FEE_GAIN,
                 c2=C2, c1=C1, r1=R1, zin=Zin, fsample=f_sample,
                 flpf1=f_LPF1, flpf2=f_LPF2,
                 noise_FEEPMB_rms=NOISE_I, noise_DAQ_rms=NOISE_DAQ, lsb=LSB):

        self.R1 = r1
        self.Zin = zin
        self.C2 = c2
        self.C1 = c1
        self.GAIN = gain
        self.A1 = self.R1 * self.Zin/(self.R1 + self.Zin)  # ohms
        self.A2 = gain / self.A1  # ohms/ohms = []
        self.R = self.R1 + self.Zin
        self.Cr = 1. + self.C1 / self.C2
        self.C = self.C1 / self.Cr
        self.ZC = self.Zin / self.Cr

        self.f_sample = fsample
        self.freq_LHPF = 1 / (self.R * self.C)
        self.freq_LPF1 = flpf1 * 2 * np.pi
        self.freq_LPF2 = flpf2 * 2 * np.pi

        self.freq_LHPFd = self.freq_LHPF / (self.f_sample * np.pi)
        self.freq_LPF1d = self.freq_LPF1 / (self.f_sample * np.pi)
        self.freq_LPF2d = self.freq_LPF2 / (self.f_sample * np.pi)
        self.coeff_blr  = self.freq_LHPFd * np.pi

        self.freq_zero = 1 / (self.R1 * self.C1)
        self.coeff_c = self.freq_zero / (self.f_sample * np.pi)

        run_number = 0 # until we decide something else, MC is run 0
        DataPMT = DB.DataPMT(run_number)

        self.coeff_blr_pmt = DataPMT.coeff_blr.values
        self.freq_LHPFd_pmt = self.coeff_blr_pmt / np.pi
        self.coeff_c_pmt = DataPMT.coeff_c.values
        self.C1_pmt = (self.coeff_blr_pmt / self.coeff_c_pmt) * (self.C2 / np.pi)
        self.R1_pmt = 1 / (self.coeff_c_pmt * self.C1_pmt * self.f_sample * np.pi)
        self.A1_pmt = self.R1_pmt * self.Zin / (self.R1_pmt + self.Zin)  # ohms
        self.A2_pmt = gain / self.A1_pmt  # ohms/ohms = []
        self.Cr_pmt = 1 + self.C1_pmt / self.C2
        self.ZC_pmt = self.Zin / self.Cr_pmt
        self.noise_FEEPMB_rms = noise_FEEPMB_rms
        self.LSB = lsb
        self.voltsToAdc = self.LSB / units.volt
        self.DAQnoise_rms = noise_DAQ_rms

    def __str__(self):
        s = """
        (C1 = {0:7.1f} nf,
         C2 = {1:7.1f} nf,
         R1 = {2:7.1f} ohm,
         Zin = {3:7.1f} ohm,
         gain = {4:7.1f} ohm,
         A1 = {5:7.4f} ohm,
         A2 = {6:7.4f},
         f_sample = {7:7.1f} MHZ,
         freq_LHPF = {8:7.2f} kHz,
         freq_LPF1 = {9:7.2f} MHZ,
         freq_LPF2 = {10:7.2f} MHZ,
         freq_LHPFd = {11:8.5f},
         coeff_blr = {12:8.5f},
         freq_LPF1d = {13:7.2f},
         freq_LPF2d = {14:7.2f},
         noise_FEEPMB_rms = {15:7.2f} muA,
         LSB = {16:7.2g} mV,
         volts to adc = {17:7.2g},
         DAQnoise_rms = {18:7.2g},
         freq_LHPFd (PMTs) = {19:s},
         coef_blr (PMTs)= {20:s},
         freq_zero = {21:7.5g},
         coeff_c = {22:7.5g},
         coeff_c (PMTs)= {23:s},
         R1 (PMTs)= {24:s} ohm,
         A1 (PMTs)= {25:s} ohm,
         A2 (PMTs)= {26:s}
        )
        """.format(self.C1               / units.nF,
                   self.C2               / units.nF,
                   self.R1               / units.ohm,
                   self.Zin              / units.ohm,
                   self.GAIN             / units.ohm,
                   self.A1               / units.ohm,
                   self.A2,
                   self.f_sample         /  units.MHZ,
                   self.freq_LHPF        / (units.kHz * 2 * np.pi),
                   self.freq_LPF1        / (units.MHZ * 2 * np.pi),
                   self.freq_LPF2        / (units.MHZ * 2 * np.pi),
                   self.freq_LHPFd,
                   self.coeff_blr,
                   self.freq_LPF1d,
                   self.freq_LPF2d,
                   self.noise_FEEPMB_rms / units.muA,
                   self.LSB              / units.mV,
                   self.voltsToAdc,
                   self.DAQnoise_rms     / units.mV,
                   self.freq_LHPFd_pmt,
                   self.coeff_blr_pmt,
                   self.freq_zero,
                   self.coeff_c,
                   self.coeff_c_pmt,
                   self.R1_pmt           / units.ohm,
                   self.A1_pmt           / units.ohm,
                   self.A2_pmt)
        return s

    def __repr__(self):
        return self.__str__()


def noise_adc(fee, signal_in_adc):
    """Equivalent Noise of the DAQ added at the output
    of the system.

    input: a signal (in units of adc counts)
           an instance of FEE class
    output: a signal with DAQ noise added
    """
    noise_daq = fee.DAQnoise_rms*v_to_adc()
    return signal_in_adc + np.random.normal(0,
                                            noise_daq,
                                            len(signal_in_adc))


def filter_sfee_lpf(sfe):
    """
    input: an instance of class Fee
    output: buttersworth parameters of the equivalent LPT FEE filter
    """
    # LPF order 1
    b1, a1 = signal.butter(1, sfe.freq_LPF1d, 'low', analog=False)
    # LPF order 4
    b2, a2 = signal.butter(4, sfe.freq_LPF2d, 'low', analog=False)
    # convolve LPF1, LPF2
    a = np.convolve(a1, a2, mode='full')
    b_aux = np.convolve(b1, b2, mode='full')
    b = sfe.GAIN * b_aux
    return b, a


def filter_fee(feep, ipmt):
    """
    input: an instance of class FEE
           ipmt = pmt number
    output: buttersworth parameters of the equivalent FEE filter
    """

    # print(feep.freq_LHPFd_pmt)
    # print('ipmt = {}, freq_LHPFd_pmt = {}'.format(ipmt,
    #                                              feep.freq_LHPFd_pmt[ipmt]))
    # high pass butterswoth filter ~1/RC
    coef = feep.freq_LHPFd_pmt[ipmt]
    A1 = feep.A1_pmt[ipmt]
    A2 = feep.A2_pmt[ipmt]
    ZC = feep.ZC_pmt[ipmt]
    if ipmt == -1:
        coef = feep.freq_LHPFd
        A1 = feep.A1
        A2 = feep.A2
        ZC = feep.ZC
    b1, a1 = signal.butter(1, coef, 'high', analog=False)
    b2, a2 = signal.butter(1, coef, 'low', analog=False)

    b0 = b2*ZC + b1*A1  # in ohms
    a0 = a1

    # LPF order 1
    b1l, a1l = signal.butter(1,
                             feep.freq_LPF1d, 'low',
                             analog=False)
    # LPF order 4
    b2l, a2l = signal.butter(4,
                             feep.freq_LPF2d, 'low',
                             analog=False)
    # convolve HPF, LPF1
    a_aux = np.convolve(a0, a1l, mode='full')
    b_aux = np.convolve(b0, b1l, mode='full')
    # convolve HPF+LPF1, LPF2
    a = np.convolve(a_aux, a2l, mode='full')
    b_aux2 = np.convolve(b_aux, b2l, mode='full')
    b = A2 * b_aux2  # in ohms

    return b, a


def filter_cleaner(feep, ipmt):
    """Clean the input signal."""
    coef = feep.coeff_c_pmt[ipmt]
    if ipmt == -1:
        coef = feep.coeff_c
    #  freq_zero = 1./(feep.R1*feep.C1)
    #  freq_zerod = freq_zero/(feep.f_sample*np.pi)
    b, a = signal.butter(1, coef, 'high', analog=False)

    return b, a


def signal_v_fee(feep, signal_i, ipmt):
    """
    input: signal_i = signal current (i = A)
           instance of class FEE
           pmt number
    output: signal_v (in volts) with effect FEE

    ++++++++++++++++++++++++++++++++++++++++++++++++
    +++++++++++ PMT+FEE NOISE ADDED HERE +++++++++++
    ++++++++++++++++++++++++++++++++++++++++++++++++

    """
    if (feep.noise_FEEPMB_rms == 0):
        noise_FEEin = np.zeros(len(signal_i))
    else:
        noise_FEEin = np.random.normal(0,
                                       feep.noise_FEEPMB_rms,
                                       len(signal_i))

    # Equivalent Noise of the FEE + PMT BASE added at the input
    # of the system to get the noise filtering effect

    b, a = filter_fee(feep, ipmt)  # b in ohms
    # filtered signal in I*R = V
    return signal.lfilter(b, a, signal_i + noise_FEEin)


def signal_v_lpf(feep, signal_in):
    """
    input: instance of class sfe and a current signal
    outputs: signal convolved with LPF in voltage
    """
    b, a = filter_sfee_lpf(feep)
    return signal.lfilter(b, a, signal_in)


def signal_clean(feep, signal_fee, ipmt):
    """
    input: signal_fee = adc, convoluted
           instance of class FEE
    output: signal_c cleaning filter passed

    ++++++++++++++++++++++++++++++++++++++++++++++++
    +++++++++++ PMT+FEE NOISE ADDED HERE +++++++++++
    ++++++++++++++++++++++++++++++++++++++++++++++++

    """
    b, a = filter_cleaner(feep, ipmt)
    return signal.lfilter(b, a, signal_fee)


def daq_decimator(f_sample1, f_sample2, signal_in):
    """Downscale the signal vector according to the
    scale defined by f_sample1 (1 GHZ) and
    f_sample2 (40 Mhz).
    Includes anti-aliasing filter
    """

    scale = int(f_sample1 / f_sample2)
    return signal.decimate(signal_in, scale, ftype='fir', zero_phase=False)
