import os
from collections import namedtuple

import tables as tb
import numpy  as np

from pytest import mark
from pytest import fixture

from .  isidora        import isidora
from .. core           import system_of_units as units
from .. core.configure import configure


@mark.slow
def test_isidora_electrons_40keV(config_tmpdir, ICDATADIR):
    # NB: avoid taking defaults for PATH_IN and PATH_OUT
    # since they are in general test-specific
    # NB: avoid taking defaults for run number (test-specific)

    PATH_IN  = os.path.join(ICDATADIR    , 'electrons_40keV_z250_RWF.h5')
    PATH_OUT = os.path.join(config_tmpdir, 'electrons_40keV_z250_CWF.h5')

    nrequired  = 2

    conf = configure('dummy invisible_cities/config/liquid_isidora.conf'.split())
    conf.update(dict(run_number   = 0,
                     files_in     = PATH_IN,
                     file_out     = PATH_OUT,
                     event_range  = (0, nrequired)))

    cnt = isidora(**conf)
    nactual = cnt.events_in
    assert nrequired == nactual

    with tb.open_file(PATH_IN,  mode='r') as h5in, \
         tb.open_file(PATH_OUT, mode='r') as h5out:
            nrow = 0

            # check events numbers & timestamps
            evts_in  = h5in .root.Run.events[:nactual].astype([('evt_number', '<i4'), ('timestamp', '<u8')])
            evts_out = h5out.root.Run.events[:nactual]
            np.testing.assert_array_equal(evts_in, evts_out)
