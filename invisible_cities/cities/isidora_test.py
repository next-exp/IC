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
from .. core.configure       import all as all_events
from .. core.testing_utils   import assert_tables_equality


@mark.slow
def test_isidora_electrons_40keV(config_tmpdir, ICDIR):
    # NB: avoid taking defaults for PATH_IN and PATH_OUT
    # since they are in general test-specific
    # NB: avoid taking defaults for run number (test-specific)

    PATH_IN = os.path.join(ICDIR, 'database/test_data/', 'electrons_40keV_z250_RWF.h5')
    PATH_OUT = os.path.join(config_tmpdir,               'electrons_40keV_z250_CWF.h5')

    nrequired  = 2

    conf = configure('dummy invisible_cities/config/isidora.conf'.split())
    conf.update(dict(run_number   = 0,
                     files_in     = PATH_IN,
                     file_out     = PATH_OUT,
                     event_range  = (0, nrequired)))

    isidora = Isidora(**conf)
    isidora.run()
    cnt = isidora.end()

    nactual = cnt.n_events_tot
    if nrequired > 0:
        assert nrequired == nactual

    with tb.open_file(PATH_IN,  mode='r') as h5in, \
         tb.open_file(PATH_OUT, mode='r') as h5out:
            nrow = 0

            # check events numbers & timestamps
            evts_in     = h5in .root.Run.events[:nactual]
            evts_out_i4 = h5out.root.Run.events[:nactual]

            evts_out_u8 = evts_out_i4.astype([('evt_number', '<u8'), ('timestamp', '<u8')])

            np.testing.assert_array_equal(evts_in, evts_out_u8)


def test_isidora_exact_result(ICDATADIR, output_tmpdir):
    file_in     = os.path.join(ICDATADIR    , "Kr83_nexus_v5_03_00_ACTIVE_7bar_3evts.RWF.h5")
    file_out    = os.path.join(output_tmpdir,                      "exact_result_isidora.h5")
    true_output = os.path.join(ICDATADIR    , "Kr83_nexus_v5_03_00_ACTIVE_7bar_3evts.BLR.h5")

    conf = configure("isidora invisible_cities/config/isidora.conf".split())
    conf.update(dict(run_number  = 0,
                     files_in    = file_in,
                     file_out    = file_out,
                     event_range = all_events))

    Isidora(**conf).run()

    tables = ( "MC/extents",  "MC/hits"   , "MC/particles", "MC/generators",
              "BLR/pmtcwf" , "BLR/sipmrwf",
              "Run/events" , "Run/runInfo")
    with tb.open_file(true_output)  as true_output_file:
        with tb.open_file(file_out) as      output_file:
            print(true_output_file)
            print(output_file)
            for table in tables:
                got      = getattr(     output_file.root, table)
                expected = getattr(true_output_file.root, table)
                assert_tables_equality(got, expected)
