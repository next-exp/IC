import numpy  as np
import pandas as pd

from hypothesis              import given
from hypothesis.strategies   import integers
from hypothesis.strategies   import lists
from hypothesis.extra.pandas import columns, data_frames

from .  dst_functions import dst_event_id_selection


@given(data_frames(columns=columns(['event'], elements=integers(min_value=-1e5, max_value=1e5))),
       lists(integers(min_value=-1e5, max_value=1e5)))
def test_dst_event_id_selection(dst, events):
    filtered_dst = dst_event_id_selection(dst, events)
    assert set(filtered_dst.event.values) == set(dst.event.values) & set(events)


def test_dst_event_id_selection_2():
    data         = {'event': [1, 1, 3, 6, 7], 'values': [3, 4, 2, 5, 6]}
    filt_data    = {'event': [1, 1, 6], 'values': [3, 4, 5]}

    df_data      = pd.DataFrame(data=data)
    df_filt_data = pd.DataFrame(data=filt_data)
    df_real_filt = dst_event_id_selection(df_data, [1, 2, 6, 10])

    assert np.all(df_real_filt.reset_index(drop=True) == df_filt_data.reset_index(drop=True))
