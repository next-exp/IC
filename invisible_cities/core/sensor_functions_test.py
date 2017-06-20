import pandas as pd
import numpy  as np

from .. database        import load_db
from . sensor_functions import convert_channel_id_to_IC_id


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
