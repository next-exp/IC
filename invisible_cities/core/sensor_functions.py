"""Functions manipulating sensors (PMTs and SiPMs)
JJGC January 2017
"""
import numpy  as np
import pandas as pd

from .. database     import load_db


def convert_channel_id_to_IC_id(data_frame, channel_ids):
    return pd.Index(data_frame.ChannelID).get_indexer(channel_ids)
