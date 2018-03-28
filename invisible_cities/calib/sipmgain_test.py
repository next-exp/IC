"""
code: sipmgain_test.py
description: test suite for sipmgain (currently just adapted from isidora_test)
author: A. Laing
IC core team: Jacek Generowicz, JJGC,
G. Martinez, J.A. Hernando, J.M Benlloch
package: invisible cities. See release notes and licence
last changed:
"""

import os

import tables as tb

from numpy.testing import assert_array_equal
from pytest        import mark

from .  sipmgain       import Sipmgain
from .. core.configure import configure


@mark.slow
@mark.parametrize("proc_opt", ('subtract_mode', 'subtract_median'))
def test_sipmgain_pulsedata(config_tmpdir, ICDATADIR, proc_opt):
    PATH_IN   = os.path.join(ICDATADIR     , 'sipmledpulsedata.h5')
    PATH_OUT  = os.path.join(config_tmpdir, 'sipmledpulsedata_HIST.h5')
    nrequired = 2

    conf = configure('dummy invisible_cities/config/sipmgain.conf'.split())
    conf.update(dict(run_number   = 4000,
                     files_in     = PATH_IN,
                     file_out     = PATH_OUT,
                     event_range  = (0, nrequired),
                     proc_mode    = proc_opt      ))

    sipmgain = Sipmgain(**conf)
    sipmgain.run()
    cnt = sipmgain.end()

    nactual = cnt.n_events_tot
    assert nrequired == nactual

    with tb.open_file(PATH_IN,  mode='r') as h5in, \
         tb.open_file(PATH_OUT, mode='r') as h5out:
        # The old format used <i4 for th event number; the new one
        # uses <u8. Casting the latter to the former allows us to
        # re-use the old test data files.
        evts_in  = h5in .root.Run.events[:nactual]
        evts_out = h5out.root.Run.events[:nactual].astype([('evt_number', '<i4'),
                                                           ('timestamp' , '<u8')])

        assert_array_equal(evts_in, evts_out)
