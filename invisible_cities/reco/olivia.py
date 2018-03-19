import sys
import os
import json

import numpy as np

from   invisible_cities.io  .hist_io   import save_histomanager_to_file
from   invisible_cities.core.configure import configure

import invisible_cities.reco.monitor_functions   as monf


data_types = ['rwf', 'pmaps']

def olivia(conf):
    print(conf)
    in_path      =     os.path.expandvars(conf.in_path)
    out_path     =     os.path.expandvars(conf.out_path)
    run_number   =                    int(conf.run_number)
    data_type    =                        conf.data_type
    histo_config =                        conf.histo_config

    if data_type not in data_types:
        print(f'Error: Data type {data_type} is not recognized.')
        print(f'Supported data types: {data_types}')
        sys.exit(2)

    with open(histo_config) as  config_file:
        config_dict = json.load(config_file)
        if   data_type == 'rwf'  :
            histo_manager = monf.fill_rwf_histos (in_path, config_dict)
        elif data_type == 'pmaps':
            histo_manager = monf.fill_pmap_histos(in_path, run_number, config_dict)
        save_histomanager_to_file(histo_manager, out_path)


if __name__ == "__main__":
    conf = configure(sys.argv).as_namespace
    olivia(conf)
