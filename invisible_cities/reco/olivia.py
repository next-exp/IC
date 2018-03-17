import sys
import glob
import os

from collections import defaultdict

import numpy as np
import json

from   invisible_cities.io .pmaps_io import load_pmaps
from   invisible_cities.io .hist_io  import save_histomanager_to_file
from   invisible_cities.evm.histos   import HistoManager

import invisible_cities.reco.histogram_functions as histf
import invisible_cities.reco.monitor_functions   as monf
import invisible_cities.database.load_db         as dbf


ICDIR = os.environ['ICDIR']

data_types = ['rwf', 'pmaps'  ]
sources    = ['kr' , 'generic']
config_files = {'pmaps_kr'      : os.path.join(ICDIR, 'config/PmapsKrConfig.json')     ,
                'pmaps_generic' : os.path.join(ICDIR, 'config/PmapsGenericConfig.json'),
                'rwf'           : os.path.join(ICDIR, 'config/RwfConfig.json')         }

def olivia():
    if len(sys.argv) < 6:
        print("Error: There are missing parameters")
        print("Usage: python RunMonitoring.py IN_PATH OUT_PATH RUN_NUMBER DATA_TYPE SOURCE\n")
        sys.exit(2)

    in_path    = sys.argv[1]
    out_path   = sys.argv[2]
    run_number = int(sys.argv[3])
    data_type  = sys.argv[4]
    source     = sys.argv[5]

    if data_type not in data_types:
        print('Error: Data type {} is not recognized.\n'.format(data_type))
        print('Supported data types: {}'.format(data_types))
        sys.exit(2)

    if source not in sources:
        print('Error: Source {} is not recognized.\n'.format(source))
        print('Supported sources: {}'.format(sources))
        sys.exit(2)

    if data_type == 'rwf':
        with open(config_files[data_type]) as config_file:
            config_dict = json.load(config_file)
        histo_manager = monf.fill_rwf_histos(in_path, config_dict)
        save_histomanager_to_file(histo_manager, out_path)

    elif data_type == 'pmaps':
        with open(config_files[data_type+'_'+source]) as config_file:
            config_dict = json.load(config_file)
        histo_manager = monf.fill_pmap_histos(in_path, run_number, config_dict)
        save_histomanager_to_file(histo_manager, out_path)

if __name__ == "__main__":
    olivia()
