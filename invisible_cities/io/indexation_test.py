import os
import tables as tb

from pytest import mark
from pytest import fixture

from .. cities.base_cities import City
from .. core  .configure   import configure

from .  hits_io import     hits_writer
from .  kdst_io import       kr_writer
from .    mc_io import mc_track_writer
from . pmaps_io import     pmap_writer


class DummyCity(City):
    def display_IO_info(self):
        "Override so it is not executed"

    def file_loop(self):
        "Override so it is not executed"

    def get_writers(self, h5out): pass
    def write_parameters(self, h5out): pass


@fixture(scope="session")
def init_city(ICDIR, config_tmpdir):
    conf = configure(('city ' + ICDIR +  'config/city.conf').split())
    file_out = os.path.join(config_tmpdir, "empty_file.h5")
    conf.update(dict(file_out = file_out))
    city = DummyCity(**conf)
    return city, file_out


@mark.parametrize("writer group node column".split(),
                  [(    hits_writer, "RECO" , "Events"  , "event"     ),
                   (      kr_writer, "DST"  , "Events"  , "event"     ),
                   (mc_track_writer, "MC"   , "MCTracks", "event_indx"),
                   (    pmap_writer, "PMAPS", "S1"      , "event"     ),
                   (    pmap_writer, "PMAPS", "S2"      , "event"     ),
                   (    pmap_writer, "PMAPS", "S2Si"    , "event"     )])
def test_table_is_indexed(init_city, writer, group, node, column):
    city, file_out = init_city
    city.get_writers = writer
    city.run()
    city.end() # or city.index_tables(), but this checks that city.end() calls city.index_tables()
    with tb.open_file(file_out, 'r') as h5out:
        table = getattr(getattr(h5out.root, group), node)
        assert getattr(table.cols, column).is_indexed
