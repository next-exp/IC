import os

from importlib import import_module
from itertools import chain

import tables as tb

from pytest import mark
from pytest import raises

from .. core.configure      import configure

online_cities = "irene dorothea sophronia esmeralda beersheba".split()
all_cities    = """beersheba berenice buffy detsim diomira dorothea esmeralda
                   eutropia hypathia irene isaura isidora phyllis sophronia
                   trude""".split()

@mark.filterwarnings("ignore::UserWarning")
@mark.parametrize("city", online_cities)
def test_city_empty_input_file(config_tmpdir, ICDATADIR, city):
    # All cities run in the online reconstruction chain must run on an
    # empty file without raising any exception

    PATH_IN  = os.path.join(ICDATADIR    , 'empty_file.h5')
    PATH_OUT = os.path.join(config_tmpdir, 'empty_output.h5')

    config_file = 'dummy invisible_cities/config/{}.conf'.format(city)
    conf = configure(config_file.split())
    conf.update(dict(files_in      = PATH_IN,
                     file_out      = PATH_OUT))

    module_name   = f'invisible_cities.cities.{city}'
    city_function = getattr(import_module(module_name), city)

    city_function(**conf)



@mark.filterwarnings("ignore::UserWarning")
@mark.parametrize("city", all_cities)
def test_city_output_file_is_compressed(config_tmpdir, ICDATADIR, city):
    file_out    = os.path.join(config_tmpdir, f"{city}_compression.h5")
    config_file = 'dummy invisible_cities/config/{}.conf'.format(city)

    conf = configure(config_file.split())
    conf.update(dict(file_out = file_out))

    module_name   = f'invisible_cities.cities.{city}'
    city_function = getattr(import_module(module_name), city)

    city_function(**conf)

    with tb.open_file(file_out) as file:
        for node in chain([file], file.walk_nodes()):
            try:
                assert (node.filters.complib   is not None and
                        node.filters.complevel > 0)

            except tb.NoSuchNodeError:
                continue


@mark.filterwarnings("ignore::UserWarning")
@mark.parametrize("city", all_cities)
def test_city_missing_detector_db(city):
    # use default config file
    config_file = 'dummy invisible_cities/config/{}.conf'.format(city)
    conf = configure(config_file.split())

    # delete the detector_db key from the config
    del conf['detector_db']

    # collect relevant city information for running
    module_name   = f'invisible_cities.cities.{city}'
    city_function = getattr(import_module(module_name), city)

    # check that the test provides the
    # required value error for missing detector_db
    with raises(ValueError, match=r"The function `(\w+)` is missing an argument `detector_db`"):
    	city_function(**conf)
