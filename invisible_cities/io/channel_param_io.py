import tables as tb
from .. core     import tbl_functions as tbl


## Suggested parameter list
## Useful for generic/default fit functions
generic_params = ["normalization", "poisson_mu"    ,
                  "pedestal"     , "pedestal_sigma",
                  "gain"         , "gain_sigma"    ,
                  "fit_limits"   , "n_gaussians_chi2" ]


def create_param_table(h5out, sensor_type, func_name, parameter_dict):

    ## If the group 'FITPARAMS' doesn't already exist, create it
    try:                       PARAM_group = getattr(h5out.root, "FITPARAMS")
    except tb.NoSuchNodeError: PARAM_group = h5out.create_group(h5out.root,
                                                                "FITPARAMS")
    ## Define a table for this fitting function
    param_table = h5out.create_table(PARAM_group,
                                     "FIT_"+sensor_type+"_"+func_name,
                                     parameter_dict,
                                     "Calibration parameters",
                                     tbl.filters("NOCOMPR"))
    return param_table


def store_fit_values(param_table, sensor_id, fit_result):

    channel = param_table.row
    channel["SensorID"] = sensor_id
    for key, param in fit_result.items():
        channel[key] = param
    channel.append()
    param_table.flush()


def make_table_dictionary(param_names, covariance=None):
    """
    returns a dictionary to be used in the
    table definition.
    """

    parameter_dict = {'SensorID' : tb.UInt32Col(pos=0)}
    for i, par in enumerate(param_names):
        parameter_dict[par] = tb.Float32Col(pos=i + 1, shape=2)

    if covariance:
        # We're saving the covariance matrix too
        cov_pos = len(param_names) + 1
        parameter_dict["covariance"] = tb.Float32Col(pos=cov_pos,
                                                     shape=covariance)

    return parameter_dict


def channel_param_writer(h5out, *, sensor_type,
                         func_name, param_names,
                         covariance=None):
    """
    Define a group, table and writer function for the ouput
    of fit parameters.
    input:
    h5out : the predefined hdf5 output file
    sensor_type : e.g. pmt, sipm or FE
    func_name : A string with the name of the fitting function being used
    param_names : A list of parameter names
    covariance : None or a tuple with the shape of the covariance matrix
    """

    parameter_dict = make_table_dictionary(param_names, covariance)

    param_table = create_param_table(h5out, sensor_type,
                                     func_name, parameter_dict)
    def store_channel_fit(sensor_id, fit_result):
        """
        input:
        sensor_id : Sensor number
        fit_result : dict with keys as parameter names
                     Fit parameters should be (value, error)
        """
        store_fit_values(param_table, sensor_id, fit_result)

    return store_channel_fit


def basic_param_reader(h5in):
    """
    Reads information from the FITPARAMS group.
    returns tuple with lists of
    table names, parameter names and the tables present
    """
    try:
        param_tables = h5in.root.FITPARAMS._f_list_nodes('Leaf')
        table_names = [ tab.name for tab in param_tables ]
        param_names = [ tab.colnames for tab in param_tables ]
        return table_names, param_names, param_tables
    except tb.NoSuchNodeError:
        print('File does not contain FITPARAMS node')
        exit()


def generator_param_reader(h5in, table_name):
    """
    Accesses the file to get the tables and
    loops over the table requested yielding
    sensor number, (parameter value dict, parameter error dict)
    """

    table_names, param_names, param_tables = basic_param_reader(h5in)

    try:
        indx = table_names.index(table_name)
        for row in param_tables[indx].iterrows():
            yield row['SensorID'], parameters_and_errors(row, param_names[indx][1:])
    except ValueError:
        print('Requested table not present')
        exit()


def subset_param_reader(h5in, table_name, param_names):

    table_names, _, param_tables = basic_param_reader(h5in)

    try:
        indx = table_names.index(table_name)
        for row in param_tables[indx].iterrows():
            yield row['SensorID'], parameters_and_errors(row, param_names)
    except ValueError:
        print('Requested table not present')
        exit()


def all_channel_value_reader(param_table, param_names):
    """
    Like subset_param_reader but with the correct
    table already extracted from file.
    """
    for row in param_table.iterrows():
        yield row['SensorID'], parameters_and_errors(row, param_names)


def single_channel_value_reader(channel, param_table, param_names):
    """
    Read the parameters for a specific channel
    assuming table already extracted
    """
    channel_params = param_table.read_where('SensorID=='+str(channel))
    if channel_params.size == 0:
        print('Sensor info not found')
        exit()
    elif channel_params.size > 1:
        print('Ambiguous call, more than one sensor entry found')
        exit()
    return parameters_and_errors(channel_params[0], param_names)


def parameters_and_errors(table_row, parameters):
    param_dict = {}
    error_dict = {}
    for pname in parameters:
        param_dict[pname] = table_row[pname][0]
        error_dict[pname] = table_row[pname][1]

    return param_dict, error_dict

