import os

import numpy  as np
import tables as tb

from numpy.testing import assert_allclose

from .. reco     import tbl_functions as tbl

from . channel_param_io import        generic_params
from . channel_param_io import      store_fit_values
from . channel_param_io import  channel_param_writer
from . channel_param_io import make_table_dictionary


def test_generic_parameters(config_tmpdir):
    filename = os.path.join(config_tmpdir, 'test_param.h5')

    outDict = {}
    with tb.open_file(filename, 'w') as dataOut:
        pWrite = channel_param_writer(dataOut, sensor_type="test",
                                      func_name="generic",
                                      param_names=generic_params)

        for i, par in enumerate(generic_params):
            outDict[par] = [i, i / 10]

        pWrite(0, outDict)

    col_list = ['SensorID'] + list(outDict.keys())
    val_list = np.concatenate(list(outDict.values()))
    with tb.open_file(filename) as dataIn:

        tblRead = dataIn.root.FITPARAMS.FIT_test_generic
        assert tblRead.nrows == 1
        assert len(tblRead.colnames) == len(generic_params) + 1
        assert tblRead.colnames == col_list
        assert_allclose(np.concatenate(list(tblRead[0])[1:]), val_list)


def test_simple_parameters_with_covariance(config_tmpdir):
    filename = os.path.join(config_tmpdir, 'test_param.h5')

    simple = ["par0", "par1", "par2"]
    cov = np.array([[0, 1, 2], [3, 4, 5]])
    outDict = {}
    with tb.open_file(filename, 'w') as dataOut:
        pWrite = channel_param_writer(dataOut, sensor_type="test",
                                      func_name="simple",
                                      param_names=simple, covariance=cov.shape)

        for i, par in enumerate(simple):
            outDict[par] = (i, i / 10)
        outDict["covariance"] = cov
        
        pWrite(0, outDict)

    with tb.open_file(filename) as dataIn:

        file_cov = dataIn.root.FITPARAMS.FIT_test_simple[0]["covariance"]
        assert_allclose(file_cov, cov)


def test_make_table_dictionary():

    param_names = ["par0", "par1", "par2"]
    
    par_dict = make_table_dictionary(param_names)
    
    # Add the sensor id to the test list
    param_names = ["SensorID"] + param_names

    assert param_names == list(par_dict.keys())


def test_store_fit_values(config_tmpdir):
    filename = os.path.join(config_tmpdir, 'test_param.h5')

    dummy_dict = {'SensorID' : tb.UInt32Col(pos=0),
                  'par0' : tb.UInt32Col(pos=1)}

    with tb.open_file(filename, 'w') as dataOut:
        PARAM_group = dataOut.create_group(dataOut.root, "testgroup")

        param_table = dataOut.create_table(PARAM_group,
                                           "testtable",
                                           dummy_dict,
                                           "test parameters",
                                           tbl.filters("NOCOMPR"))
        
        store_fit_values(param_table, 0, {'par0' : 22})

    with tb.open_file(filename) as dataIn:

        tblRead = dataIn.root.testgroup.testtable
        assert tblRead.nrows == 1
        assert len(tblRead.colnames) == len(dummy_dict)
        assert tblRead.colnames == list(dummy_dict.keys())

