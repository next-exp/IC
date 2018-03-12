import os
import tables as tb

from pytest import mark
from pytest import fixture

from .. liquid_cities.components import city
from .. core  .configure   import configure

from . hits_io   import     hits_writer
from . kdst_io   import       kr_writer
from . mcinfo_io import  mc_info_writer
from . pmaps_io  import     pmap_writer


@city
def writer_test_city(writer, file_out, files_in, event_range):
    with tb.open_file(file_out, 'w') as h5out:
        writer(h5out)

    def get_writers     (self, h5out): pass
    def write_parameters(self, h5out): pass


@fixture(scope="session")
def init_city(ICDIR, config_tmpdir):
    conf = configure(('city ' + ICDIR +  'config/city.conf').split())
    file_out = os.path.join(config_tmpdir, "empty_file.h5")
    conf.update(dict(file_out = file_out))
    city = DummyCity(**conf)
    return city, file_out


@mark.parametrize("         writer  group      node      column        thing".split(),
                  [(   hits_writer, "RECO" , "Events"  , "event"     , "hits"),
                   (     kr_writer, "DST"  , "Events"  , "event"     , "kr"  ),
                   (mc_info_writer, "MC"   , "extents" , "evt_number", "mc"  ),
                   (   pmap_writer, "PMAPS", "S1"      , "event"     , "s1"  ),
                   (   pmap_writer, "PMAPS", "S2"      , "event"     , "s2"  ),
                   (   pmap_writer, "PMAPS", "S2Si"    , "event"     , "s2si")])
def test_table_is_indexed(tmpdir_factory, writer, group, node, column, thing):
    tmpdir = tmpdir_factory.mktemp('indexation')
    file_out = os.path.join(tmpdir, f"empty_table_containing_{thing}.h5")
    writer_test_city(writer=writer, file_out=file_out, files_in='dummy')
    with tb.open_file(file_out, 'r') as h5out:
        table = getattr(getattr(h5out.root, group), node)
        assert getattr(table.cols, column).is_indexed
