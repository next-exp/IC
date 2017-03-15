"""
code: pmap_io_test.py
"""
import os
import tables as tb
import numpy  as np

from invisible_cities.reco.pmap_io           import pmap_writer, S12, S2Si
from invisible_cities.core.system_of_units_c import units
from invisible_cities.reco.pmaps_functions   import read_pmaps


def test_pmap_writer_mc(config_tmpdir):
    # Check that PMAPS tables are written correctly

    PMP_file_name = os.path.join(str(config_tmpdir), 'test_pmaps_mc.h5')

    run_number = 0
    S1 = S12({0:(np.random.rand(5),
                 np.random.rand(5))})
    S2 = S12({0:(np.random.rand(5),
                 np.random.rand(5))})
    Si = S2Si({0:[(1, np.random.rand(5)),
                  (2, np.random.rand(5)),
                  (3, np.random.rand(5))]})

    # Automatic closing example
    with pmap_writer(PMP_file_name) as write:
        write(0, S1, S2, Si)

    # Manual closing example
    write = pmap_writer(PMP_file_name)
    write(0, S1, S2, Si)
    write.close()

    s1df, s2df, s2sidf = read_pmaps(PMP_file_name)
    np.testing.assert_allclose(s1df.time.values, S1[0][0])
    np.testing.assert_allclose(s2df.time.values, S2[0][0])
    dim = len(Si[0][0][1])
    assert len(s2sidf) == len(Si[0]) * dim
    np.testing.assert_allclose(s2sidf.ene.values[0:dim], Si[0][0][1])
