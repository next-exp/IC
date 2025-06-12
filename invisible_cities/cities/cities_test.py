import os

from importlib import import_module
from itertools import chain

import tables as tb

from pytest import mark
from pytest import raises

from .. core.configure     import configure
from .. core.testing_utils import ignore_warning


online_cities  = "irene dorothea sophronia esmeralda beersheba isaura".split()
mc_cities      = "buffy detsim diomira hypathia".split()
other_cities   = "berenice isidora phyllis trude eutropia".split()
all_cities     = sorted(online_cities + mc_cities + other_cities)
all_cities_with_event_range = sorted(set(all_cities).difference(set(["eutropia"])))


@ignore_warning.no_config_group
@ignore_warning.no_mc_tables
@ignore_warning.not_kdst
@ignore_warning.str_length
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


@ignore_warning.no_config_group
@ignore_warning.not_kdst
@ignore_warning.str_length
@mark.parametrize("city", all_cities_with_event_range)
def test_city_null_event_range(config_tmpdir, ICDATADIR, city):
    # All cities must run with event_range = 0
    PATH_OUT = os.path.join(config_tmpdir, 'empty_output.h5')

    config_file = 'dummy invisible_cities/config/{}.conf'.format(city)
    conf = configure(config_file.split())
    conf.update(dict( file_out = PATH_OUT
                     , event_range = 0))

    module_name   = f'invisible_cities.cities.{city}'
    city_function = getattr(import_module(module_name), city)

    city_function(**conf)


@ignore_warning.no_config_group
@ignore_warning.not_kdst
@ignore_warning.str_length
@mark.parametrize("city", all_cities)
def test_city_output_file_is_compressed(config_tmpdir, ICDATADIR, city):
    file_out    = os.path.join(config_tmpdir, f"{city}_compression.h5")
    config_file = 'dummy invisible_cities/config/{}.conf'.format(city)

    conf = configure(config_file.split())
    conf.update(dict( file_out    = file_out
                    , event_range = 0))

    module_name   = f'invisible_cities.cities.{city}'
    city_function = getattr(import_module(module_name), city)

    city_function(**conf)

    checked_nodes = 0
    with tb.open_file(file_out) as file:
        for node in chain([file], file.walk_nodes()):
            try:
                assert (node.filters.complib   is not None and
                        node.filters.complevel > 0)
                checked_nodes += 1

            except tb.NoSuchNodeError:
                continue

    # ensure that we didn't pass the test because there were no nodes checked
    assert checked_nodes > 2


@ignore_warning.no_config_group
@ignore_warning.not_kdst
@ignore_warning.str_length
@mark.parametrize("city", all_cities)
def test_city_output_contains_configuration(config_tmpdir, city):
    file_out    = os.path.join(config_tmpdir, f"{city}_configuration.h5")
    config_file = 'dummy invisible_cities/config/{}.conf'.format(city)

    conf = configure(config_file.split())
    conf.update(dict( file_out    = file_out
                    , event_range = 0))

    module_name   = f'invisible_cities.cities.{city}'
    city_function = getattr(import_module(module_name), city)

    city_function(**conf)

    with tb.open_file(file_out) as file:
        assert "config" in file.root
        assert city     in file.root.config
        table = getattr(file.root.config, city)
        assert len(table[:]) > 0


@ignore_warning.no_config_group
def test_cities_carry_on_configuration_from_previous_ones(ICDATADIR, config_tmpdir):
    rwf_file   = os.path.join(    ICDATADIR, "electrons_40keV_ACTIVE_10evts_RWF.h5")
    pmaps_file = os.path.join(config_tmpdir, f"test_cities_carry_on_configuration_pmaps.h5")
    kdst_file  = os.path.join(config_tmpdir, f"test_cities_carry_on_configuration_kdst.h5")

    conf_irene = configure("dummy invisible_cities/config/irene.conf".split())
    conf_irene.update(dict( files_in    = rwf_file
                          , file_out    = pmaps_file
                          , event_range = 0))

    conf_dorothea = configure("dummy invisible_cities/config/dorothea.conf".split())
    conf_dorothea.update(dict( files_in = pmaps_file
                             , file_out = kdst_file
                             , event_range = 0))

    for city, conf in (("irene", conf_irene), ("dorothea", conf_dorothea)):
        module_name   = f'invisible_cities.cities.{city}'
        city_function = getattr(import_module(module_name), city)
        city_function(**conf)

    # now check that the second output file contains the configuration from both cities
    with tb.open_file(kdst_file) as file:
        assert "config" in file.root

        for city in "irene dorothea".split():
            assert city in file.root.config
            table = getattr(file.root.config, city)
            assert len(table[:]) > 0


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
