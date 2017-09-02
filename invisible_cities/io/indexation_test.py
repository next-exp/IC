import os
import tables as tb

from pytest import mark

from . hits_io import     hits_writer
from . kdst_io import       kr_writer
from .   mc_io import mc_track_writer
from . pmap_io import     pmap_writer


@mark.parametrize("writer content group node column".split(),
                  [(    hits_writer, "hits"   , "RECO" , "Events"  , "event"     ),
                   (      kr_writer, "kr"     , "DST"  , "Events"  , "event"     ),
                   (mc_track_writer, "mctrk"  , "MC"   , "MCTracks", "event_indx"),
                   (    pmap_writer, "S1pmaps", "PMAPS", "S1"      , "event"     ),
                   (    pmap_writer, "S2pmaps", "PMAPS", "S2"      , "event"     ),
                   (    pmap_writer, "Sipmaps", "PMAPS", "S2Si"    , "event"     )])
def test_hits_table_is_indexed(config_tmpdir, writer, content, group, node, column):
    outputfilename = os.path.join(config_tmpdir, f"empty_{content}_file.h5")

    with tb.open_file(outputfilename, 'w') as h5out:
        writer(h5out)
        table = getattr(getattr(h5out.root, group), node)
        assert getattr(table.cols, column).is_indexed
