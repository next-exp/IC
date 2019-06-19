import os
import tables as tb

from importlib import import_module
from pytest import mark
from .. core.configure      import configure

cities = "irene dorothea penthesilea esmeralda".split()

@mark.parametrize("city", cities)
def test_city_empty_input_file(config_tmpdir, ICDATADIR, city):
    # All cities run in Canfranc must run on an empty file
    # without raising any exception

    PATH_IN  = os.path.join(ICDATADIR    , 'empty_file.h5')
    PATH_OUT = os.path.join(config_tmpdir, 'empty_output.h5')

    config_file = 'dummy invisible_cities/config/{}.conf'.format(city)
    conf = configure(config_file.split())
    conf.update(dict(files_in      = PATH_IN,
                     file_out      = PATH_OUT))

    module_name   = f'invisible_cities.cities.{city}'
    city_function = getattr(import_module(module_name), city)

    city_function(**conf)
