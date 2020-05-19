import os

import tables as tb
import numpy  as np

from .. core.configure import configure
from .. core.testing_utils  import assert_tables_equality

from . hypathia import hypathia


def test_hypathia_exact_result(ICDATADIR, output_tmpdir):
    file_in     = os.path.join(ICDATADIR    ,      "Kr83_nexus_v5_03_00_ACTIVE_7bar_3evts.MCRD.h5")
    file_out    = os.path.join(output_tmpdir,                          "exact_result_hypathia.h5")
    true_output = os.path.join(ICDATADIR    , "Kr83_nexus_v5_03_00_ACTIVE_7bar_3evts.hypathia.h5")

    conf = configure("hypathia invisible_cities/config/hypathia.conf".split())
    conf.update(dict(run_number   = -6340,
                     files_in     = file_in,
                     file_out     = file_out,
                     event_range  = (0, 2)))

    # Set a specific seed because we want the result to be
    # repeatible. Go back to original state after running.
    original_random_state = np.random.get_state()
    np.random.seed(123456789)
    hypathia(**conf)
    np.random.set_state(original_random_state)

    tables = (  "PMAPS/S1"         ,   "PMAPS/S2"        , "PMAPS/S2Si"     ,
                "PMAPS/S1Pmt"      ,   "PMAPS/S2Pmt"     ,
                  "Run/events"     ,     "Run/runInfo"   ,
              "Trigger/events"     , "Trigger/trigger"   ,
              "Filters/s12_indices", "Filters/empty_pmap")
    with tb.open_file(true_output)  as true_output_file:
        with tb.open_file(file_out) as      output_file:
            for table in tables:
                got      = getattr(     output_file.root, table)
                expected = getattr(true_output_file.root, table)
                assert_tables_equality(got, expected)
