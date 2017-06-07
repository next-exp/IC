"""
code: cecilia_test.py
description: test suite for trigger city
author: P. Ferrario
IC core team: Jacek Generowicz, JJGC,
G. Martinez, J.A. Hernando, J.M Benlloch
package: invisible cities. See release notes and licence
"""
import os
import tables as tb
import numpy as np

from pytest import mark, fixture

from invisible_cities.cities.cecilia import Cecilia, CECILIA
import invisible_cities.reco.peak_functions_c as cpf
import invisible_cities.database.load_db as db

@fixture(scope='module')
def conf_file_name(config_tmpdir):
    # Specifies a name for a MC configuration file. Also, default number
    # of events set to 1.

    conf_file_name = str(config_tmpdir.join('cecilia_test.conf'))
    Cecilia.write_config_file(conf_file_name,
                              PATH_OUT = str(config_tmpdir),
                              COMPRESSION = "ZLIB4",
                              FIRST_EVT = 0,
                              TR_CHANNELS = "0 1 2 3 4 5 6 7 8 9 10 11",
                              MIN_NUMB_CHANNELS = 5,
                              MIN_HEIGHT = 15,
                              MAX_HEIGHT = 1000,
                              MIN_CHARGE = 3000,
                              MAX_CHARGE = 20000,
                              MIN_WIDTH = 4000,
                              MAX_WIDTH = 12000,
                              DATA_MC_RATIO = 0.8)
    return conf_file_name

def test_command_line_trigger_krypton(conf_file_name, config_tmpdir, ICDIR):
    # NB: avoid taking defaults for PATH_IN and PATH_OUT
    # since they are in general test-specific
    # NB: avoid taking defaults for run number (test-specific)

    PATH_IN = os.path.join(ICDIR,
                           'database/test_data/',
                           'kr_cwf.root.h5')
    PATH_OUT = os.path.join(str(config_tmpdir),
                            'kr_trigwf.h5')

    nevts, n_evts_run = CECILIA(['CECILIA',
                                 '-c', conf_file_name,
                                 '-i', PATH_IN,
                                 '-o', PATH_OUT,
                                 '-n', '4',
                                 '-r', '0'])
    if nevts != -1:
        assert nevts == n_evts_run

def test_filtering_krypton(conf_file_name, config_tmpdir, ICDIR):
    # This test checks that the events that pass the filter have
    # at least 1 PMT among those used for trigger with the correct
    # values for charge
    
    PATH_IN = os.path.join(ICDIR,
                           'database/test_data/',
                           'kr_cwf.root.h5')
    PATH_OUT = os.path.join(str(config_tmpdir),
                            'kr_trigwf.h5')

    nevts, n_evts_run = CECILIA(['CECILIA',
                                 '-c', conf_file_name,
                                 '-i', PATH_IN,
                                 '-o', PATH_OUT,
                                 '-n', '4',
                                 '-r', '0'])

    config_path = os.path.join(str(config_tmpdir),
                                conf_file_name)
    min_charge = 0.
    max_charge = 0.
    min_height = 0.
    min_n_pmt = 0
    tr_channels = []

    with open (config_path, "r") as conf_file:
        param = conf_file.readlines()
        for l in param:
            ll=l.split(" ")
            if len(ll)>1:
                if ll[0] == 'MIN_CHARGE':
                    min_charge = float(ll[1])
                elif ll[0] =='MAX_CHARGE':
                    max_charge = float(ll[1])
                elif ll[0] == 'MIN_HEIGHT':
                    min_height = float(ll[1])
                elif ll[0] == 'MIN_NUMB_CHANNELS':
                    min_n_pmt = float(ll[1])
                elif ll[0] == 'TR_CHANNELS':
                    for i in range(1,len(ll)):
                        tr_channels.append(int(ll[i]))
    
    try:
        with tb.open_file(PATH_OUT, "r+") as h5in:
                pmttrigwf  = h5in.root.pmttrigwf
                nevts, npmts, pmt_wndw_length = pmttrigwf.shape
                n_correct = 0              
                for evt in range(nevts):
                    for pmt in range(npmts):
                        if (pmt in tr_channels):                          
                            found = False
                            pk_content,pk_indx=cpf.wfzs(pmttrigwf[evt, pmt].astype(np.double), threshold=min_height)
                            pks = cpf.find_S12(pk_content, pk_indx)
                
                            for value in pks.values():
                                charge = np.sum(value[1])
                                if charge > min_charge and charge < max_charge:
                                    found = True
                                    
                            if found:
                                n_correct += 1
                                break
                            
        assert n_correct ==  nevts      
                        
    except:
        print('error')
        raise
        
        
