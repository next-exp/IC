import tables as tb
import numpy  as np
import pandas as pd

from pytest import mark

from .. core.random_sampling import NoiseSampler as SiPMsNoiseSampler
from .. sierpe               import blr
from .. database             import load_db

from .                   import wfm_functions as wfm
from .  sensor_functions import convert_channel_id_to_IC_id
from .  sensor_functions import simulate_pmt_response
from .. reco             import calib_sensors_functions as csf


def test_cwf_blr(dbnew, electron_MCRD_file):
    """Check that CWF -> (deconv) (RWF) == BLR within 1 %. """

    run_number    = 0
    DataPMT       = load_db.DataPMT(dbnew, run_number)
    pmt_active    = np.nonzero(DataPMT.Active.values)[0].tolist()
    coeff_blr     = abs(DataPMT.coeff_blr.values)
    coeff_c       = abs(DataPMT.coeff_c .values)
    adc_to_pes    = abs(DataPMT.adc_to_pes.values)
    single_pe_rms = abs(DataPMT.Sigma.values)
    thr_trigger   = 5

    with tb.open_file(electron_MCRD_file, 'r') as h5in:
        event = 0
        pmtrd = h5in.root.pmtrd
        dataPMT, BLR = simulate_pmt_response(event, pmtrd, adc_to_pes, single_pe_rms)
        ZWF = csf.means(dataPMT[:, :28000]) - dataPMT
        CWF = np.array(tuple(map(blr.deconvolve_signal, ZWF[pmt_active] ,
                                 coeff_c              , coeff_blr       ,
                                 np.repeat(thr_trigger, len(pmt_active)))))

        diff = wfm.compare_cwf_blr(cwf         = [CWF],
                                   pmtblr      = [BLR],
                                   event_list  = [0],
                                   window_size = 500)
        assert diff[0] < 1

@mark.slow
def test_sipm_noise_sampler(dbnew, electron_MCRD_file):
    """This test checks that the number of SiPMs surviving a hard energy
        cut (50 pes) is  small (<10). The test exercises the full
       construction of the SiPM vectors as well as the noise suppression.
    """

    run_number = 0
    DataSiPM = load_db.DataSiPM(dbnew, run_number)
    sipm_adc_to_pes = DataSiPM.adc_to_pes.values.astype(np.double)

    cal_min = 13
    cal_max = 19
    # the average calibration constant is 16 see diomira_nb in Docs
    sipm_noise_cut = 30 # in pes. Should kill essentially all background

    max_sipm_with_signal = 10

    with tb.open_file(electron_MCRD_file, 'r') as e40rd:
        event = 0

        NEVENTS_DST, NSIPM, SIPMWL = e40rd.root.sipmrd.shape

        assert NSIPM == 1792
        assert SIPMWL == 1200

        assert np.mean(sipm_adc_to_pes[sipm_adc_to_pes>0]) > cal_min
        assert np.mean(sipm_adc_to_pes[sipm_adc_to_pes>0]) < cal_max

        sipms_thresholds = sipm_noise_cut *  sipm_adc_to_pes
        noise_sampler = SiPMsNoiseSampler(dbnew, run_number, SIPMWL, True)

        # signal in sipm with noise
        sipmrwf = e40rd.root.sipmrd[event] + noise_sampler.sample()
        # zs waveform
        sipmzs = wfm.noise_suppression(sipmrwf, sipms_thresholds)
        n_sipm = 0
        for j in range(sipmzs.shape[0]):
            if np.sum(sipmzs[j] > 0):
                n_sipm+=1
        assert n_sipm < max_sipm_with_signal

def test_channel_id_to_IC_id():
    data_frame = pd.DataFrame({'SensorID' : [2,9,4,1,3],
                               'ChannelID': [2,8,5,7,6]})

    assert np.array_equal(
        convert_channel_id_to_IC_id(data_frame, [2,6,5]),
        np.array                               ([0,4,2]))


def test_channel_id_to_IC_id_with_real_data(dbnew):
    pmt_df = load_db.DataPMT(dbnew, 0)
    assert np.array_equal(
        convert_channel_id_to_IC_id(pmt_df, pmt_df.ChannelID.values),
                                            pmt_df.SensorID .values)
