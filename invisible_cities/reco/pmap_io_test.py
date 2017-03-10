"""
code: pmap_io_test.py
"""
import os
import tables as tb
import numpy  as np

from invisible_cities.reco.pmap_io           import PMapWriter
from invisible_cities.core.system_of_units_c import units
from invisible_cities.core.core_functions    import trange
from invisible_cities.reco.pmaps_functions   import read_pmaps


def test_pmap_writer_mc(config_tmpdir):
    # Check that PMAPS tables are written correctly

    PMP_file_name = os.path.join(str(config_tmpdir), 'test_pmaps_mc.h5')

    run_number = 0
    S1 = {0:(np.array(trange(5), dtype=np.float),
             np.array(trange(5), dtype=np.float))}
    S2 = S1
    S2Si = {0:[(1, np.array(trange(5), dtype=np.float)),
               (2, np.array(trange(5), dtype=np.float)),
               (3, np.array(trange(5), dtype=np.float))]}

    with tb.open_file(PMP_file_name, "w") as pmap_file:
        pmw = PMapWriter(pmap_file)
        pmw.store_pmaps(0, S1, S2, S2Si)
        pmw.flush()

    s1df, s2df, s2sidf = read_pmaps(PMP_file_name)
    np.testing.assert_array_equal(s1df.time.values, S1[0][0])
    np.testing.assert_array_equal(s2df.time.values, S2[0][0])
    dim = len(S2Si[0][0][1])
    assert len(s2sidf) == len(S2Si[0]) * dim
    np.testing.assert_array_equal(s2sidf.ene.values[0:dim], S2Si[0][0][1])
