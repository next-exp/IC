import os
import tables as tb
import pandas as pd
from pytest import mark

from .. cities.components  import city
from .. core.configure     import OneOrManyFiles
from .. core.configure     import EventRangeType
from .. core.testing_utils import ignore_warning

from . hits_io   import     hits_writer
from . kdst_io   import       kr_writer
from . pmaps_io  import     pmap_writer

from . dst_io    import  df_writer

from typing import Callable
from typing import Any


# type annotations needed
@city
def writer_test_city( writer      : Callable
                    , file_out    : str
                    , files_in    : OneOrManyFiles
                    , event_range : EventRangeType
                    , detector_db : str
                    ):
    with tb.open_file(file_out, 'w') as h5out:
        writer(h5out)


def _df_writer(h5out):
    df = pd.DataFrame(columns=['event', 'some_value'], dtype=int)
    return df_writer(h5out, df, 'DUMMY', 'dummy', columns_to_index=['event'])


@ignore_warning.no_config_group
@ignore_warning.str_length
@mark.parametrize("         writer  group      node      column   thing".split(),
                  [(   hits_writer, "RECO" , "Events"  , "event", "hits"),
                   (     kr_writer, "DST"  , "Events"  , "event", "kr"  ),
                   (   pmap_writer, "PMAPS", "S1"      , "event", "s1"  ),
                   (   pmap_writer, "PMAPS", "S2"      , "event", "s2"  ),
                   (   pmap_writer, "PMAPS", "S2Si"    , "event", "s2si"),
                   (    _df_writer, "DUMMY", "dummy"   , "event", 'df'  )])
def test_table_is_indexed(tmpdir_factory, ICDATADIR, writer, group, node, column, thing):
    tmpdir      = tmpdir_factory.mktemp('indexation')
    file_out    = os.path.join(tmpdir, f"empty_table_containing_{thing}.h5")
    dummy_files = os.path.join(ICDATADIR, "*.h5")
    writer_test_city(writer=writer, file_out=file_out, files_in=dummy_files, detector_db = 'new')
    with tb.open_file(file_out, 'r') as h5out:
        table = getattr(getattr(h5out.root, group), node)
        assert getattr(table.cols, column).is_indexed
