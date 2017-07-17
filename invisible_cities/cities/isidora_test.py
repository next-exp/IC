"""
code: isidora_test.py
description: test suite for isidora
author: J.J. Gomez-Cadenas
IC core team: Jacek Generowicz, JJGC,
G. Martinez, J.A. Hernando, J.M Benlloch
package: invisible cities. See release notes and licence
last changed: 01-12-2017
"""

import os
from collections import namedtuple

import tables as tb
import numpy  as np

from pytest import mark
from pytest import fixture

from .  isidora import Isidora
from .. core                 import system_of_units as units
from .. core.configure       import configure


@mark.slow
def test_isidora_electrons_40keV(config_tmpdir, ICDIR):
    # NB: avoid taking defaults for PATH_IN and PATH_OUT
    # since they are in general test-specific
    # NB: avoid taking defaults for run number (test-specific)

    PATH_IN = os.path.join(ICDIR, 'database/test_data/', 'electrons_40keV_z250_RWF.h5')
    PATH_OUT = os.path.join(config_tmpdir,               'electrons_40keV_z250_CWF.h5')

    nrequired  = 2

    conf = configure('dummy invisible_cities/config/isidora.conf'.split()).as_dict
    conf.update(dict(run_number = 0,
                     filesin   = PATH_IN,
                     file_out   = PATH_OUT,
                     nmax       = nrequired))

    isidora = Isidora(**conf)
    cnt = isidora.run()

    nactual = cnt.counter_value('n_events_tot')
    if nrequired > 0:
        assert nrequired == nactual
        assert nrequired == cnt.counter_value('nmax')

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
