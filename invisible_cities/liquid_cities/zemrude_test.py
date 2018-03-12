import os

import tables as tb
import numpy  as np

from pytest import mark

from .  zemrude            import zemrude
from .. core.configure     import configure
from .. core.testing_utils import assert_array_equal

def test_zemrude_sipmdarkcurrent(config_tmpdir, ICDATADIR):
    PATH_IN   = os.path.join(ICDATADIR    , 'sipmdarkcurrentdata.h5' )
    PATH_OUT  = os.path.join(config_tmpdir, 'sipmdarkcurrentdata_HIST.h5')
    nrequired = 2

    conf = configure('dummy invisible_cities/config/liquid_zemrude.conf'.split())
    conf.update(dict(run_number  = 4000,
                     files_in    = PATH_IN,
                     file_out    = PATH_OUT,
                     event_range = (0, nrequired)))

    cnt = zemrude(**conf)
    assert cnt.events_in == nrequired

    with tb.open_file(PATH_IN , mode='r') as h5in, \
         tb.open_file(PATH_OUT, mode='r') as h5out:

        evts_in  = h5in .root.Run.events[:nrequired].astype([('evt_number', '<i4'), ('timestamp', '<u8')])
        evts_out = h5out.root.Run.events[:nrequired]
        assert_array_equal(evts_in, evts_out)
