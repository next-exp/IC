"""
code: sipmpdf_test.py
description: test suite for sipmPDF (currently just adapted from isidora_test)
author: A. Laing
IC core team: Jacek Generowicz, JJGC,
G. Martinez, J.A. Hernando, J.M Benlloch
package: invisible cities. See release notes and licence
last changed:
"""


import os

import tables as tb
import numpy  as np

from pytest import mark

from .  sipmpdf        import Sipmpdf
from .. core.configure import configure


@mark.slow
def test_sipmpdf_electrons_40keV(config_tmpdir, ICDATADIR):
    # NB: avoid taking defaults for PATH_IN and PATH_OUT
    # since they are in general test-specific
    # NB: avoid taking defaults for run number (test-specific)

    PATH_IN  = os.path.join(ICDATADIR    , 'electrons_40keV_z250_RWF.h5' )
    PATH_OUT = os.path.join(config_tmpdir, 'electrons_40keV_z250_HIST.h5')

    nrequired  = 2

    conf = configure('dummy invisible_cities/config/sipmpdf.conf'.split())
    conf.update(dict(run_number   = 0,
                     files_in     = PATH_IN,
                     file_out     = PATH_OUT,
                     event_range  = (0, nrequired)))

    sipmpdf = Sipmpdf(**conf)
    sipmpdf.run()
    cnt = sipmpdf.end()

    nactual = cnt.n_events_tot
    if nrequired > 0:
        assert nrequired == nactual

    with tb.open_file(PATH_IN,  mode='r') as h5in, \
         tb.open_file(PATH_OUT, mode='r') as h5out:
            nrow = 0

            # check events numbers & timestamps
            evts_in     = h5in .root.Run.events[:nactual]
            evts_out_u8 = h5out.root.Run.events[:nactual]
            # The old format used <i4 for th event number; the new one
            # uses <u8. Casting the latter to the former allows us to
            # re-use the old test data files.
            evts_out_i4 = evts_out_u8.astype([('evt_number', '<i4'), ('timestamp', '<u8')])
            np.testing.assert_array_equal(evts_in, evts_out_i4)
