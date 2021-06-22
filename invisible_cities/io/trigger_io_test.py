import os

import numpy  as np
import tables as tb

from numpy     .testing        import assert_allclose
from hypothesis                import given
from hypothesis.strategies     import integers
from hypothesis.strategies     import lists
from hypothesis.strategies     import booleans
from hypothesis.strategies     import composite

from ..core.testing_utils      import assert_dataframes_equal
from ..io.dst_io               import load_dst
from . trigger_io              import trigger_dst_writer
from .. reco.trigger_functions import TriggerInfo


@composite
def trigger_value(draw):
    uints = integers(min_value=0                     , max_value=np.iinfo(np.uint32).max)
    ints  = integers(min_value=np.iinfo(np.int32).min, max_value=np.iinfo(np.int32).max)

    # Event and PMT ID
    evt_id = draw(uints)
    pmt_id = draw(uints)

    # Peak time, q, width and height
    time   = draw(uints)
    q      = draw(uints)
    width  = draw(uints)
    height = draw(uints)

    # Trigger validity flags
    valid_q    = draw(booleans())
    valid_w    = draw(booleans())
    valid_h    = draw(booleans())
    valid_peak = draw(booleans())
    valid_all  = all([valid_q, valid_w, valid_h, valid_peak])

    # Baseline and wvf max. height
    baseline   = draw(ints)
    max_height = draw(ints)

    # Number of coincidences, closest pmt and tbin
    n_coinc      = draw(ints)
    closest_time = draw(ints)
    closest_pmt  = draw(ints)

    trigger = [evt_id  , pmt_id    , time   , q           , width      , height
              ,valid_q , valid_w   , valid_h, valid_peak  , valid_all
              ,baseline, max_height, n_coinc, closest_time, closest_pmt        ]

    return TriggerInfo(*trigger)


@composite
def trigger_values(draw):
    size  = draw(integers(min_value=1, max_value=10))
    return draw(lists(trigger_value(), min_size=size, max_size=size))


@given(triggers=trigger_values())
def test_trigger_dst_writer(config_tmpdir, triggers):
    output_file = os.path.join(config_tmpdir, "test_trigger_dst.h5")

    with tb.open_file(output_file, 'w') as h5out:
        write = trigger_dst_writer(h5out)
        write(triggers)

    trigger_dst = load_dst(output_file, group='Trigger', node='DST')

    col_names = ["event"     , "pmt"       , "trigger_time" , "q"
                ,"width"     , "height"    , "valid_q"      , "valid_w"
                ,"valid_h"   , "valid_peak", "valid_all"    , "baseline"
                ,"max_height", "n_coinc"   , "closest_ttime", "closest_pmt"]

    assert np.all([tname == cname for tname, cname in zip(trigger_dst.columns.values, col_names)])

    # Check values stored
    for i, row in trigger_dst.iterrows():
        assert_allclose(triggers[i], list(row.values))
