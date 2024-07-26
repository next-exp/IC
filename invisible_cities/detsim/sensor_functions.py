"""Functions manipulating sensors (PMTs and SiPMs)
JJGC January 2017
"""
import numpy  as np
import pandas as pd

from .. sierpe            import fee as FE
from .. sierpe            import low_frequency_noise as lfn
from .. reco              import wfm_functions as wfm


def convert_channel_id_to_IC_id(data_frame, channel_ids):
    return pd.Index(data_frame.ChannelID).get_indexer(channel_ids)


def charge_fluctuation(signal, single_pe_rms):
    """Simulate the fluctuation of the pe before noise addition
    produced by each photoelectron
    """
    if single_pe_rms == 0:
        ## Protection for some versions of numpy etc
        return signal

    ## We need to convert to float to get accuracy here
    sig_fl   = signal.astype(float)
    non_zero = sig_fl > 0
    sigma    = np.sqrt(sig_fl[non_zero]) * single_pe_rms
    sig_fl[non_zero] = np.random.normal(sig_fl[non_zero], sigma)
    ## This fluctuation can't give negative signal
    sig_fl[sig_fl < 0] = 0
    return sig_fl


def simulate_pmt_response(event, pmtrd, adc_to_pes, pe_resolution, detector_db='new', run_number = 0):
    """ Full simulation of the energy plane response
    Input:
     1) extensible array pmtrd
     2) event_number
    returns:
    array of raw waveforms (RWF) obtained by convoluting pmtrd with the PMT
    front end electronics (LPF, HPF filters)
    array of BLR waveforms (only decimation)
    """
    # Single Photoelectron class
    spe = FE.SPE()
    # FEE, with noise PMT
    fee  = FE.FEE(detector_db, run_number,
                  noise_FEEPMB_rms=FE.NOISE_I, noise_DAQ_rms=FE.NOISE_DAQ)
    # Low frequency noise
    buffer_length = int(FE.f_sample * pmtrd.shape[2] / FE.f_mc)
    lowFreq = lfn.low_frequency_noise(detector_db, run_number, buffer_length)

    NPMT = pmtrd.shape[1]
    RWF  = []
    BLRX = []
    for pmt in range(NPMT):
        # normalize calibration constants from DB to MC value
        cc = adc_to_pes[pmt] / FE.ADC_TO_PES
        # signal_i in current units
        # fluctuating charge according to 1pe sigma from calibration.
        pe_rms = pe_resolution[pmt]
        signal_i = FE.spe_pulse_from_vector(spe,
                                            charge_fluctuation(pmtrd[event, pmt], pe_rms),
                                            norm=cc)
        # Decimate (DAQ decimation)
        signal_d = FE.daq_decimator(FE.f_mc, FE.f_sample, signal_i)
        # Effect of FEE and transform to adc counts
        signal_fee = FE.signal_v_fee(fee, signal_d, pmt) * FE.v_to_adc()
        # add noise daq including the low frequency noise
        signal_daq = FE.noise_adc(fee, signal_fee) - lowFreq(pmt)
        # signal blr is just pure MC decimated by adc in adc counts
        signal_blr = FE.signal_v_lpf(fee, signal_d) * FE.v_to_adc()
        # raw waveform stored with negative sign and offset
        RWF.append(FE.OFFSET - signal_daq)
        # blr waveform stored with positive sign and no offset
        BLRX.append(signal_blr)
    return np.array(RWF), np.array(BLRX)


def simulate_sipm_response(sipmrd, sipms_noise_sampler, sipm_adc_to_pes, pe_resolution):
    """Add noise to the sipms with the NoiseSampler class and return
    the noisy waveform (in adc)."""

    ## Fluctuate according to charge resolution
    sipm_fl = np.array(tuple(map(charge_fluctuation, sipmrd, pe_resolution)))

    # return total signal in adc counts + noise sampled from pdf spectra
    return wfm.to_adc(sipm_fl, sipm_adc_to_pes) + sipms_noise_sampler.sample()
