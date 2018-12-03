import os

import tables as tb
import numpy  as np

from pytest import mark

from .  trude              import trude
from .. core.configure     import configure
from .. core.configure     import       all as all_events
from .. core.testing_utils import assert_array_equal
from .. core.testing_utils import assert_tables_equality


@mark.slow
@mark.parametrize("proc_opt", ('subtract_mode', 'subtract_median'))
def test_trude_pulsedata(config_tmpdir, ICDATADIR, proc_opt):
    PATH_IN   = os.path.join(ICDATADIR    , 'sipmledpulsedata.h5')
    PATH_OUT  = os.path.join(config_tmpdir, 'sipmledpulsedata_HIST.h5')
    nrequired = 2

    conf = configure('dummy invisible_cities/config/trude.conf'.split())
    conf.update(dict(run_number  = 4000,
                     files_in    = PATH_IN,
                     file_out    = PATH_OUT,
                     event_range = (0, nrequired),
                     proc_mode   = proc_opt      ))

    cnt = trude(**conf)
    assert cnt.events_in == nrequired

    with tb.open_file(PATH_IN , mode='r') as h5in, \
         tb.open_file(PATH_OUT, mode='r') as h5out:

        evts_in  = h5in .root.Run.events[:nrequired].astype([('evt_number', '<i4'), ('timestamp', '<u8')])
        evts_out = h5out.root.Run.events[:nrequired]
        assert_array_equal(evts_in, evts_out)


@mark.parametrize("proc_opt", ("subtract_mode", "subtract_median"))
def test_trude_exact_result(ICDATADIR, output_tmpdir, proc_opt):
    file_in     = os.path.join(ICDATADIR    ,                  "sipmledpulsedata.h5")
    file_out    = os.path.join(output_tmpdir,  f"exact_result_trude_{proc_opt}.h5")
    true_output = os.path.join(ICDATADIR    , f"sipmledpulsedata_hist_{proc_opt}.h5")

    conf = configure("trude invisible_cities/config/trude.conf".split())
    conf.update(dict(run_number  = 4832,
                     files_in    = file_in,
                     file_out    = file_out,
                     proc_mode   = proc_opt,
                     event_range = all_events))

    trude(**conf)

    tables = ("HIST/sipm_dark", "HIST/sipm_dark_bins",
              "HIST/sipm_spe" , "HIST/sipm_spe_bins" ,
               "Run/events"   ,  "Run/runInfo"       )
    with tb.open_file(true_output)  as true_output_file:
        with tb.open_file(file_out) as      output_file:
            for table in tables:
                got      = getattr(     output_file.root, table)
                expected = getattr(true_output_file.root, table)
                assert_tables_equality(got, expected)
