import os

import numpy  as np
import tables as tb

from numpy.testing import assert_allclose
from hypothesis                import given
from hypothesis.strategies     import integers
from hypothesis.strategies     import lists
from hypothesis.strategies     import booleans
from hypothesis.strategies     import composite

from ..core.testing_utils import assert_dataframes_equal
from ..io.dst_io          import load_dst
from . trigger_io         import trigger_dst_writer


@composite
def trigger_values(draw):
    size  = draw(integers(min_value=1, max_value=10))
    elements = []
    for _ in range(6):
        elements.append(draw(lists(integers(min_value=0, max_value=np.iinfo(np.uint32).max),
                                   min_size=size, max_size=size)))
    for _ in range(4):
        elements.append(draw(lists(booleans(), min_size=size, max_size=size)))
    for _ in range(5):
        elements.append(draw(lists(integers(min_value=np.iinfo(np.int32).min, max_value=np.iinfo(np.int32).max),
                                   min_size=size, max_size=size)))
    return elements


@given(triggers=trigger_values())
def test_trigger_dst_writer(config_tmpdir, triggers):
    output_file = os.path.join(config_tmpdir, "test_trigger_dst.h5")

    trigger_save = [list(a) for a in zip(*triggers)]
    with tb.open_file(output_file, 'w') as h5out:
        write = trigger_dst_writer(h5out)
        write(trigger_save)

    trigger_dst = load_dst(output_file, group='Trigger', node='DST')

    col_names = ["event"     , "pmt"       , "trigger_time" , "q"
                ,"width"     , "height"    ,  "valid_q"     , "valid_w"
                ,"valid_h"   , "valid_peak", "valid_all"    , "baseline"
                ,"max_height", "n_coinc"   , "closest_ttime", "closest_pmt"]

    assert np.all([tname == cname for tname, cname in zip(trigger_dst.columns.values, col_names)])

    # Check values stored, the if is needed because the valid_all condition is done in the writer, not on the input trigger
    for i, colname in enumerate(col_names):
        if colname == 'valid_all':
            assert_allclose(np.all(triggers[6:10], axis=0), trigger_dst.loc[:, colname])
        elif i>10: assert_allclose(triggers[i-1]          , trigger_dst.loc[:, colname])
        else     : assert_allclose(triggers[i  ]          , trigger_dst.loc[:, colname])
