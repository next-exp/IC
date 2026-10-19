import os
import numpy  as np
import tables as tb
import pandas as pd
import time   as tm

from . dst_io              import load_dst
from .  hits_io            import hits_writer
from .. types.ic_types     import NN

from .. core.testing_utils import assert_dataframes_close


def test_hits_writer_output_nodes(config_tmpdir, Th228_hits):
    output_file   = os.path.join(config_tmpdir, "test_hits.h5")
    original_hits = pd.read_hdf(Th228_hits, "/RECO/Events")

    with tb.open_file(output_file, 'w') as h5out:
        write = hits_writer(h5out, "THIS_GROUP", "THAT_NODE")
        write(original_hits)

        assert "THIS_GROUP" in h5out.root
        assert "THAT_NODE"  in h5out.root.THIS_GROUP


def test_hits_writer(config_tmpdir, Th228_hits):
    output_file   = os.path.join(config_tmpdir, "test_hits.h5")
    original_hits = pd.read_hdf(Th228_hits, "/RECO/Events")

    with tb.open_file(output_file, 'w') as h5out:
        write = hits_writer(h5out, "RECO", "Events")
        write(original_hits)

    read_hits = load_dst(output_file, group = "RECO", node = "Events")
    assert_dataframes_close(read_hits, original_hits)
