import os

import tables as tb
import numpy  as np

from numpy.testing import assert_array_equal

from pytest import mark

from .  phyllis            import phyllis
from .. core.configure     import configure
from .. core.testing_utils import assert_tables_equality
from .. core.testing_utils import ignore_warning
from .. types.symbols      import all_events
from .. types.symbols      import PMTCalibMode


@ignore_warning.no_config_group
@mark.parametrize("proc_opt", PMTCalibMode)
def test_phyllis_pulsedata(config_tmpdir, ICDATADIR, proc_opt):
    PATH_IN   = os.path.join(ICDATADIR    , 'pmtledpulsedata.h5')
    PATH_OUT  = os.path.join(config_tmpdir, 'pmtledpulsedata_HIST.h5')
    nrequired = 2

    conf = configure('dummy invisible_cities/config/phyllis.conf'.split())
    conf.update(dict(run_number   = 4000,
                     files_in     = PATH_IN,
                     file_out     = PATH_OUT,
                     event_range  = (0, nrequired),
                     proc_mode    = proc_opt      ))

    cnt = phyllis(**conf)
    assert cnt.events_in == nrequired

    with tb.open_file(PATH_IN,  mode='r') as h5in, \
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
@mark.parametrize("proc_opt", PMTCalibMode)
def test_phyllis_exact_result(ICDATADIR, output_tmpdir, proc_opt):
    file_in     = os.path.join(ICDATADIR    ,                       "pmtledpulsedata.h5")
    file_out    = os.path.join(output_tmpdir, f"exact_result_phyllis_{proc_opt.name}.h5")
    true_output = os.path.join(ICDATADIR    , f"pmtledpulsedata_hist_{proc_opt.name}.h5")

    conf = configure("phyllis invisible_cities/config/phyllis.conf".split())
    conf.update(dict(run_number  = 4819,
                     files_in    = file_in,
                     file_out    = file_out,
                     proc_mode   = proc_opt,
                     event_range = all_events))

    phyllis(**conf)

    tables = ("HIST/pmt_dark", "HIST/pmt_dark_bins",
              "HIST/pmt_spe" , "HIST/pmt_spe_bins" ,
               "Run/events"  ,  "Run/runInfo"      )
    with tb.open_file(true_output)  as true_output_file:
        with tb.open_file(file_out) as      output_file:
            for table in tables:
                got      = getattr(     output_file.root, table)
                expected = getattr(true_output_file.root, table)
                assert_tables_equality(got, expected)
