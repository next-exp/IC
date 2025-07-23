import os

import numpy  as np
import tables as tb

from pytest import fixture
from pytest import raises

from numpy.testing import assert_allclose

from .. core     import tbl_functions as tbl

from . channel_param_io import              generic_params
from . channel_param_io import            store_fit_values
from . channel_param_io import        channel_param_writer
from . channel_param_io import       make_table_dictionary
from . channel_param_io import          basic_param_reader
from . channel_param_io import      generator_param_reader
from . channel_param_io import         subset_param_reader
from . channel_param_io import single_channel_value_reader


@fixture(scope="session")
def fake_channel_data(config_tmpdir):
    filename = os.path.join(config_tmpdir, 'fake_channel_data.h5')

    out_dict = {}
    val_list = []
    n_rows = 5
    with tb.open_file(filename, 'w') as data_out:
        pWrite = channel_param_writer(data_out, sensor_type="test",
                                      func_name="generic",
                                      param_names=generic_params)

        for sens in range(n_rows):
            for i, par in enumerate(generic_params):
                out_dict[par] = [i, (i + sens) / 10]

            pWrite(sens, out_dict)
            val_list.append(list(out_dict.values()))

    val_list = np.array(val_list)
    return filename, n_rows, out_dict, val_list


def test_basic_param_reader_raises(config_tmpdir):
    filename = os.path.join(config_tmpdir, "test_basic_param_reader.h5")
    with tb.open_file(filename, "w"): pass

    with tb.open_file(filename) as file:
        with raises(tb.NoSuchNodeError):
            basic_param_reader(file)


def test_basic_param_reader(fake_channel_data):
    filename, n_rows, out_dict, val_list = fake_channel_data
    col_list = ['SensorID'] + list(out_dict.keys())
    with tb.open_file(filename) as data_in:
        tbl_names, param_names, tbls = basic_param_reader(data_in)
        assert len(tbls) == 1
        assert tbls[0].nrows == n_rows
        assert len(param_names[0]) == len(col_list)
        assert param_names[0] == col_list
        all_values = [ list(x)[1:] for x in tbls[0][:] ]
        assert_allclose(all_values, val_list)


def test_generator_param_reader(fake_channel_data):
    filename, n_rows, out_dict, val_list = fake_channel_data
    with tb.open_file(filename) as data_in:
        counter = 0
        for sens, (vals, errs) in generator_param_reader(data_in, 'FIT_test_generic'):
            assert sens == counter
            assert len(vals) == len(errs) == len(out_dict)
            assert_allclose(val_list[sens, :, 0], np.array(list(vals.values())))
            assert_allclose(val_list[sens, :, 1], np.array(list(errs.values())))
            counter += 1
        assert counter == n_rows


def test_generator_param_reader_raises(fake_channel_data):
    filename, *_ = fake_channel_data
    with tb.open_file(filename) as file:
        with raises(ValueError):
            tuple(generator_param_reader(file, "this_table_does_not_exist")) # consume iterator


def test_subset_param_reader(fake_channel_data):
    filename, n_rows, _, val_list = fake_channel_data
    with tb.open_file(filename) as file:
        values_and_errors = tuple(subset_param_reader(file, "FIT_test_generic", ("gain",)))
        assert len(values_and_errors) == n_rows
        for got, expected in zip(values_and_errors, val_list[:, 4]):
            assert np.isclose(got[1][0]["gain"], expected[0])
            assert np.isclose(got[1][1]["gain"], expected[1])


def test_subset_param_reader_raises(fake_channel_data):
    filename, *_ = fake_channel_data
    with tb.open_file(filename) as file:
        with raises(ValueError):
            tuple(subset_param_reader(file, "this_table_does_not_exist", ())) # consume iterator


def test_single_channel_value_reader(fake_channel_data):
    filename, *_ = fake_channel_data
    with tb.open_file(filename) as file:
        params_table   = file.root.FITPARAMS.FIT_test_generic
        params, errors = single_channel_value_reader(0, params_table, generic_params)
        assert len(params) == 8
        assert len(errors) == 8


def test_single_channel_value_reader_wrong_id(fake_channel_data):
    filename, *_ = fake_channel_data
    with tb.open_file(filename) as file:
        params_table   = file.root.FITPARAMS.FIT_test_generic
        with raises(ValueError):
            single_channel_value_reader(-1, params_table, generic_params)


def test_simple_parameters_with_covariance(config_tmpdir):
    filename = os.path.join(config_tmpdir, 'test_param_cov.h5')

    simple = ["par0", "par1", "par2"]
    cov = np.array([[0, 1, 2], [3, 4, 5]])
    out_dict = {}
    with tb.open_file(filename, 'w') as data_out:
        pWrite = channel_param_writer(data_out, sensor_type="test",
                                      func_name="simple",
                                      param_names=simple, covariance=cov.shape)

        for i, par in enumerate(simple):
            out_dict[par] = (i, i / 10)
        out_dict["covariance"] = cov

        pWrite(0, out_dict)

    with tb.open_file(filename) as data_in:
        file_cov = data_in.root.FITPARAMS.FIT_test_simple[0]["covariance"]
        assert_allclose(file_cov, cov)


def test_make_table_dictionary():
    param_names = ["par0", "par1", "par2"]

    par_dict = make_table_dictionary(param_names)

    # Add the sensor id to the test list
    param_names = ["SensorID"] + param_names

    assert param_names == list(par_dict.keys())


def test_store_fit_values(config_tmpdir):
    filename = os.path.join(config_tmpdir, 'test_param_fit_values.h5')

    dummy_dict = make_table_dictionary(['par0'])

    with tb.open_file(filename, 'w') as data_out:
        PARAM_group = data_out.create_group(data_out.root, "testgroup")

        param_table = data_out.create_table(PARAM_group,
                                           "testtable",
                                           dummy_dict,
                                           "test parameters",
                                           tbl.filters("NOCOMPR"))

        store_fit_values(param_table, 0, {'par0' : 22})

    with tb.open_file(filename) as data_in:

        tblRead = data_in.root.testgroup.testtable
        assert tblRead.nrows == 1
        assert len(tblRead.colnames) == len(dummy_dict)
        assert tblRead.colnames == list(dummy_dict.keys())
