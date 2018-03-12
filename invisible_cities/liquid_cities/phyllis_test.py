import os

import tables as tb
import numpy  as np

from pytest import mark

from .  phyllis            import phyllis
from .. core.configure     import configure
from .. core.testing_utils import assert_array_equal


@mark.parametrize("proc_opt", ('gain', 'gain_mau', 'gain_nodeconv'))
def test_phyllis_pulsedata(config_tmpdir, ICDATADIR, proc_opt):
    PATH_IN   = os.path.join(ICDATADIR    , 'pmtledpulsedata.h5')
    PATH_OUT  = os.path.join(config_tmpdir, 'pmtledpulsedata_HIST.h5')
    nrequired = 2

    conf = configure('dummy invisible_cities/config/liquid_phyllis.conf'.split())
    conf.update(dict(run_number   = 4000,
                     files_in     = PATH_IN,
                     file_out     = PATH_OUT,
                     event_range  = (0, nrequired),
                     proc_mode    = proc_opt      ))

    cnt = phyllis(**conf)
    assert cnt.events_in == nrequired

    with tb.open_file(PATH_IN,  mode='r') as h5in, \
         tb.open_file(PATH_OUT, mode='r') as h5out:

        evts_in  = h5in .root.Run.events[:nrequired].astype([('evt_number', '<i4'), ('timestamp', '<u8')])
        evts_out = h5out.root.Run.events[:nrequired]
        assert_array_equal(evts_in, evts_out)
