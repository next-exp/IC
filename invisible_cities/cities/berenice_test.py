import os

import tables as tb
import numpy  as np

from numpy.testing import assert_array_equal

from .  berenice           import berenice
from .. core.configure     import configure
from .. core.testing_utils import assert_tables_equality
from .. core.testing_utils import ignore_warning
from .. types.symbols      import all_events


@ignore_warning.no_config_group
def test_berenice_sipmdarkcurrent(config_tmpdir, ICDATADIR):
    PATH_IN   = os.path.join(ICDATADIR    , 'sipmdarkcurrentdata.h5' )
    PATH_OUT  = os.path.join(config_tmpdir, 'sipmdarkcurrentdata_HIST.h5')
    nrequired = 2

    conf = configure('dummy invisible_cities/config/berenice.conf'.split())
    conf.update(dict(run_number  = 4000,
                     files_in    = PATH_IN,
                     file_out    = PATH_OUT,
                     event_range = (0, nrequired)))

    cnt = berenice(**conf)
    assert cnt.events_in == nrequired

    with tb.open_file(PATH_IN , mode='r') as h5in, \
         tb.open_file(PATH_OUT, mode='r') as h5out:

        evts_in  = h5in .root.Run.events[:nrequired]
        evts_out = h5out.root.Run.events[:nrequired]
        assert_array_equal(evts_in, evts_out)

        assert 'Sensors' in h5out.root
        ch_in_pmt   = np.array(h5in .root.Sensors.DataPMT [:])
        ch_out_pmt  = np.array(h5out.root.Sensors.DataPMT [:])
        ch_in_sipm  = np.array(h5in .root.Sensors.DataSiPM[:])
        ch_out_sipm = np.array(h5out.root.Sensors.DataSiPM[:])
        assert np.all(ch_in_pmt  ==  ch_out_pmt)
        assert np.all(ch_in_sipm == ch_out_sipm)


@ignore_warning.no_config_group
def test_berenice_exact_result(ICDATADIR, output_tmpdir):
    file_in     = os.path.join(ICDATADIR    ,             "sipmdarkcurrentdata.h5")
    file_out    = os.path.join(output_tmpdir,            "exact_result_berenice.h5")
    true_output = os.path.join(ICDATADIR    , "sipmdarkcurrentdata_hist_liquid.h5")

    conf = configure("berenice invisible_cities/config/berenice.conf".split())
    conf.update(dict(run_number  = 4821,
                     files_in    = file_in,
                     file_out    = file_out,
                     event_range = all_events))

    berenice(**conf)

    tables = ("HIST/median", "HIST/median_bins",
              "HIST/mode"  , "HIST/mode_bins"  ,
              "HIST/adc"   , "HIST/adc_bins"   ,
               "Run/events",  "Run/runInfo"    )
    with tb.open_file(true_output)  as true_output_file:
        with tb.open_file(file_out) as      output_file:
            for table in tables:
                got      = getattr(     output_file.root, table)
                expected = getattr(true_output_file.root, table)
                assert_tables_equality(got, expected)
