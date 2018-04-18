import tables as tb
from .. reco     import tbl_functions as tbl


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
