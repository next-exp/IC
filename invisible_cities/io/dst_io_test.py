import os
import numpy as np
import tables as tb

from numpy.testing import assert_allclose

from ..types.ic_types      import xy
from ..reco.dst_functions  import load_dst
from ..evm.event_model     import Cluster
from ..evm.event_model     import Hit
from ..evm.event_model     import HitCollection
from . dst_io              import hits_writer


def test_hits_writer(config_tmpdir, hits_toy_data):
    output_file = os.path.join(config_tmpdir, "test_hits.h5")

    _, (npeak, nsipm, x, y, xstd, ystd, z, q, e) = hits_toy_data

    with tb.open_file(output_file, 'w') as h5out:
        write = hits_writer(h5out)
        hits = HitCollection(-1, -1)
        for i in range(len(x)):
            c = Cluster(q[i], xy(x[i], y[i]), xy(xstd[i], ystd[i]), nsipm[i])
            h = Hit(npeak[i], c, z[i], e[i])
            hits.hits.append(h)
        write(hits)

    dst = load_dst(output_file, group = "RECO", node = "Events")
    assert_allclose(npeak, dst.npeak.values)
    assert_allclose(nsipm, dst.nsipm.values)
    assert_allclose(x    , dst.X    .values)
    assert_allclose(y    , dst.Y    .values)
    assert_allclose(np.sqrt(xstd), dst.Xrms .values)
    assert_allclose(np.sqrt(ystd), dst.Yrms .values)
    assert_allclose(z    , dst.Z    .values)
    assert_allclose(q    , dst.Q    .values)
    assert_allclose(e    , dst.E    .values)
