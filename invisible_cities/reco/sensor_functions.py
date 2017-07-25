"""Functions manipulating sensors (PMTs and SiPMs)
JJGC January 2017
"""
import numpy  as np
import pandas as pd

from ..sierpe             import fee as FE
from .                    import wfm_functions as wfm

def convert_channel_id_to_IC_id(data_frame, channel_ids):
    return pd.Index(data_frame.ChannelID).get_indexer(channel_ids)


def simulate_pmt_response(event, pmtrd, adc_to_pes, run_number = 0):
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
    fee  = FE.FEE(run_number,
                  noise_FEEPMB_rms=FE.NOISE_I, noise_DAQ_rms=FE.NOISE_DAQ)
    NPMT = pmtrd.shape[1]
    RWF  = []
    BLRX = []
    for pmt in range(NPMT):
        # normalize calibration constants from DB to MC value
        cc = adc_to_pes[pmt] / FE.ADC_TO_PES
        # signal_i in current units
        signal_i = FE.spe_pulse_from_vector(spe, pmtrd[event, pmt])
        # Decimate (DAQ decimation)
        signal_d = FE.daq_decimator(FE.f_mc, FE.f_sample, signal_i)
        # Effect of FEE and transform to adc counts
        signal_fee = FE.signal_v_fee(fee, signal_d, pmt) * FE.v_to_adc()
        # add noise daq
        signal_daq = cc * FE.noise_adc(fee, signal_fee)
        # signal blr is just pure MC decimated by adc in adc counts
        signal_blr = cc * FE.signal_v_lpf(fee, signal_d) * FE.v_to_adc()
        # raw waveform stored with negative sign and offset
        RWF.append(FE.OFFSET - signal_daq)
        # blr waveform stored with positive sign and no offset
        BLRX.append(signal_blr)
    return np.array(RWF), np.array(BLRX)

def simulate_sipm_response(event, sipmrd, sipms_noise_sampler, sipm_adc_to_pes):
    """Add noise to the sipms with the NoiseSampler class and return
    the noisy waveform (in adc)."""
    # add noise (in PES) to true waveform
    dataSiPM = sipmrd[event] + sipms_noise_sampler.Sample()
    # return total signal in adc counts
    return wfm.to_adc(dataSiPM, sipm_adc_to_pes)
