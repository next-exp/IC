import os
import tables as tb
import numpy  as np
import pandas as pd
from pytest import mark

from .. core                 import system_of_units as units

from .. reco     import tbl_functions as tbl
from .. reco     import wfm_functions as wfm
from .. sierpe   import fee as FEE
from .. sierpe   import blr
from .. database import load_db
from . sensor_functions import convert_channel_id_to_IC_id
from . sensor_functions import simulate_pmt_response


def test_cwf_blr(electron_MCRD_file):
    """Check that CWF -> (deconv) (RWF) == BLR within 1 %. """

    run_number = 0
    DataPMT = load_db.DataPMT(run_number)
    pmt_active = np.nonzero(DataPMT.Active.values)[0].tolist()
    channel_id = DataPMT.ChannelID.values
    coeff_blr = abs(DataPMT.coeff_blr.values)
    coeff_c = abs(DataPMT.coeff_c .values)
    adc_to_pes = abs(DataPMT.adc_to_pes.values)

    with tb.open_file(electron_MCRD_file, 'r') as h5in:
        event = 0
        _, pmtrd, _ = tbl.get_rd_vectors(h5in)
        dataPMT, BLR = simulate_pmt_response(event, pmtrd, adc_to_pes)
        RWF = dataPMT.astype(np.int16)

        CWF = blr.deconv_pmt(RWF,
                             coeff_c,
                             coeff_blr,
                             pmt_active,
                             n_baseline=28000,
                             thr_trigger=5)

        diff = wfm.compare_cwf_blr(cwf         = [CWF],
                                   pmtblr      = [BLR],
                                   event_list  = [0],
                                   window_size = 500)
        assert diff[0] < 1


def test_channel_id_to_IC_id():
    data_frame = pd.DataFrame({'SensorID' : [2,9,4,1,3],
                               'ChannelID': [2,8,5,7,6]})

    assert np.array_equal(
        convert_channel_id_to_IC_id(data_frame, [2,6,5]),
        np.array                               ([0,4,2]))


def test_channel_id_to_IC_id_with_real_data():
    pmt_df = load_db.DataPMT(0)
    assert np.array_equal(
        convert_channel_id_to_IC_id(pmt_df, pmt_df.ChannelID.values),
                                            pmt_df.SensorID .values)
