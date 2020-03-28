import os

import tables as tb
import numpy  as np

from pytest import mark

from .  isidora            import isidora
from .. core.configure     import all as all_events
from .. core.configure     import configure
from .. core.testing_utils import assert_tables_equality

@mark.slow
def test_isidora_electrons_40keV(config_tmpdir, ICDATADIR):
    # NB: avoid taking defaults for PATH_IN and PATH_OUT
    # since they are in general test-specific
    # NB: avoid taking defaults for run number (test-specific)

    PATH_IN  = os.path.join(ICDATADIR    , 'electrons_40keV_z25_RWF.h5')
    PATH_OUT = os.path.join(config_tmpdir, 'electrons_40keV_z25_CWF.h5')

    nrequired  = 2

    conf = configure('dummy invisible_cities/config/isidora.conf'.split())
    conf.update(dict(run_number   = 0,
                     files_in     = PATH_IN,
                     file_out     = PATH_OUT,
                     event_range  = (0, nrequired)))

    cnt = isidora(**conf)
    nactual = cnt.events_in
    assert nrequired == nactual

    with tb.open_file(PATH_IN,  mode='r') as h5in, \
         tb.open_file(PATH_OUT, mode='r') as h5out:
            # check events numbers & timestamps
            evts_in  = h5in .root.Run.events[:nactual].astype([('evt_number', '<i4'), ('timestamp', '<u8')])
            evts_out = h5out.root.Run.events[:nactual]
            np.testing.assert_array_equal(evts_in, evts_out)


def test_isidora_exact_result(ICDATADIR, output_tmpdir):
    file_in     = os.path.join(ICDATADIR                                     ,
                               "Kr83_nexus_v5_03_00_ACTIVE_7bar_3evts.RWF.h5")
    file_out    = os.path.join(output_tmpdir, "exact_result_isidora.h5")
    true_output = os.path.join(ICDATADIR                                     ,
                               "Kr83_nexus_v5_03_00_ACTIVE_7bar_3evts.BLR.h5")

    conf = configure("isidora invisible_cities/config/isidora.conf".split())
    conf.update(dict(run_number  = -6340,
                     files_in    = file_in,
                     file_out    = file_out,
                     event_range = all_events))

    isidora(**conf)

    ## tables = ( "MC/extents",  "MC/hits"   , "MC/particles", "MC/generators",
    ##           "BLR/pmtcwf" , "BLR/sipmrwf",
    ##           "Run/events" , "Run/runInfo")
    tables = ("BLR/pmtcwf" , "BLR/sipmrwf",
              "Run/events" , "Run/runInfo")
    with tb.open_file(true_output)  as true_output_file:
        with tb.open_file(file_out) as      output_file:
            print(true_output_file)
            print(output_file)
            for table in tables:
                got      = getattr(     output_file.root, table)
                expected = getattr(true_output_file.root, table)
                assert_tables_equality(got, expected)
