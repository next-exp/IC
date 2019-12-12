import os

import numpy  as np
import tables as tb

from numpy.testing import assert_allclose
from pytest        import            mark

from . rwf_io      import rwf_writer


@mark.parametrize("group_name", ('root', 'RD', 'BLR'))
def test_rwf_writer(config_tmpdir, group_name):

    nevt       =       3
    nsensor    =       3
    nsample    =      10
    table_name = 'testwf'

    ofile      = os.path.join(config_tmpdir, 'testRWF.h5')

    test_data  = np.random.randint(10, size = (nevt, nsensor, nsample))

    with tb.open_file(ofile, "w") as h5out:
        rwf_writer_ = rwf_writer(h5out,
                                 group_name      = group_name,
                                 table_name      = table_name,
                                 n_sensors       =    nsensor,
                                 waveform_length =    nsample)

        for sens in test_data:
            rwf_writer_(sens)

    with tb.open_file(ofile) as h5test:
        if group_name is 'root':
            group = h5test.root
        else:
            group = getattr(h5test.root, group_name)


        assert table_name in group

        table = getattr(group, table_name)
        assert table.shape == (nevt, nsensor, nsample)
        assert_allclose(test_data, table.read())
