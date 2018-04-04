import os
from   collections import defaultdict

import numpy as np

from hypothesis import given

from .. reco             import monitor_functions   as monf
from .. reco             import histogram_functions as histf
from .. database         import load_db             as dbf
from .. core             import system_of_units     as units

from   .. evm.pmaps_test import pmaps
from   .. evm.pmaps_test import sensor_responses


@given(pmaps())
def test_fill_pmap_var_1d(pmaps):
    var_dict      = defaultdict(list)
    (s1s, s2s), _ = pmaps
    data_sipm     = dbf.DataSiPM(4670)

    monf.fill_pmap_var_1d(s1s, var_dict, "S1", DataSiPM=None     )
    monf.fill_pmap_var_1d(s2s, var_dict, "S2", DataSiPM=data_sipm)

    assert var_dict['S1_Number'][-1] == len(s1s)
    assert var_dict['S2_Number'][-1] == len(s2s)

    for i, speak in enumerate(s1s):
        assert     var_dict['S1_Energy'][i] == speak.total_energy
        assert     var_dict['S1_Width'] [i] == speak.width / units.mus
        assert     var_dict['S1_Height'][i] == speak.height
        assert     var_dict['S1_Charge'][i] == speak.total_charge
        assert     var_dict['S1_Time']  [i] == speak.time_at_max_energy / units.mus

    counter = 0
    for i, speak in enumerate(s2s):
        assert     var_dict['S2_Energy']  [i] == speak.total_energy
        assert     var_dict['S2_Width']   [i] == speak.width / units.mus
        assert     var_dict['S2_Height']  [i] == speak.height
        assert     var_dict['S2_Energy']  [i] == speak.total_energy
        assert     var_dict['S2_Charge']  [i] == speak.total_charge
        assert     var_dict['S2_Time']    [i] == speak.time_at_max_energy / units.mus
        assert     var_dict['S2_SingleS1'][i] == len(s1s)
        nsipm = len(speak.sipms.ids)
        assert     var_dict['S2_NSiPM']   [i] == nsipm
        if len(s1s) == 1:
            assert var_dict['S2_SingleS1_Energy'][i] == s1s[0].total_energy
        assert np.allclose(var_dict['S2_QSiPM'] [counter:counter + nsipm], speak.sipms.sum_over_times                  )
        assert np.allclose(var_dict['S2_IdSiPM'][counter:counter + nsipm], speak.sipms.ids                             )
        assert np.allclose(var_dict['S2_XSiPM'] [counter:counter + nsipm], data_sipm.X.values[speak.sipms.ids].tolist())
        assert np.allclose(var_dict['S2_YSiPM'] [counter:counter + nsipm], data_sipm.Y.values[speak.sipms.ids].tolist())
        counter += nsipm


@given(pmaps())
def test_fill_pmap_var_2d(pmaps):
    var_dict      = defaultdict(list)
    (s1s, s2s), _ = pmaps
    data_sipm     = dbf.DataSiPM(4670)

    monf.fill_pmap_var_1d(s1s, var_dict, "S1",      DataSiPM=None)
    monf.fill_pmap_var_1d(s2s, var_dict, "S2", DataSiPM=data_sipm)
    monf.fill_pmap_var_2d(     var_dict, 'S1')
    monf.fill_pmap_var_2d(     var_dict, 'S2')

    for i, speak in enumerate(s1s):
        assert var_dict['S1_Energy_S1_Width'] [0, i] == speak.total_energy
        assert var_dict['S1_Energy_S1_Width'] [1, i] == speak.width / units.mus
        assert var_dict['S1_Energy_S1_Height'][0, i] == speak.total_energy
        assert var_dict['S1_Energy_S1_Height'][1, i] == speak.height
        assert var_dict['S1_Energy_S1_Charge'][0, i] == speak.total_energy
        assert var_dict['S1_Energy_S1_Charge'][1, i] == speak.total_charge
        assert var_dict['S1_Time_S1_Energy']  [0, i] == speak.time_at_max_energy /units.mus
        assert var_dict['S1_Time_S1_Energy']  [1, i] == speak.total_energy

    counter = 0
    for i, speak in enumerate(s2s):
        assert     var_dict['S2_Energy_S2_Width'] [0, i] == speak.total_energy
        assert     var_dict['S2_Energy_S2_Width'] [1, i] == speak.width / units.mus
        assert     var_dict['S2_Energy_S2_Height'][0, i] == speak.total_energy
        assert     var_dict['S2_Energy_S2_Height'][1, i] == speak.height
        assert     var_dict['S2_Energy_S2_Charge'][0, i] == speak.total_energy
        assert     var_dict['S2_Energy_S2_Charge'][1, i] == speak.total_charge
        assert     var_dict['S2_Time_S2_Energy']  [0, i] == speak.time_at_max_energy /units.mus
        assert     var_dict['S2_Time_S2_Energy']  [1, i] == speak.total_energy
        if len(s1s) == 1:
            assert var_dict['S2_Energy_S1_Energy'][0, i] == speak.total_energy
            assert var_dict['S2_Energy_S1_Energy'][1, i] == s1s[0].total_energy
        sipm_ids = speak.sipms.ids
        assert np.allclose(var_dict['S2_XYSiPM'][0, counter:counter + len(sipm_ids)], data_sipm.X.values[speak.sipms.ids].tolist())
        assert np.allclose(var_dict['S2_XYSiPM'][1, counter:counter + len(sipm_ids)], data_sipm.Y.values[speak.sipms.ids].tolist())
        counter += len(sipm_ids)


def test_pmap_bins():
    test_dict = {'S1_Energy_bins'   : [   0,    10, 2],
                 'S1_Width_bins'    : [   0,     6, 3],
                 'S1_Time_bins'     : [   0,     5, 4],
                 'S2_Energy_bins'   : [   0, 15000, 2],
                 'S2_Height_bins'   : [   0, 15000, 2],
                 'S2_XSiPM_bins'    : [-100,   100, 4],
                 'S2_YSiPM_bins'    : [-100,   100, 4],
                 'S2_Time_bins'     : [   5,    10, 4],

                 'S1_Energy_labels' : ['S1 energy (pes)'],
                 'S1_Width_labels'  : [ 'S1 width (mus)'],
                 'S1_Time_labels'   : [  'S1 time (mus)'],
                 'S2_Energy_labels' : ['S2 energy (pes)'],
                 'S2_Height_labels' : ['S2 energy (pes)'],
                 'S2_XSiPM_labels'  : [         'X (mm)'],
                 'S2_YSiPM_labels'  : [         'Y (mm)'],
                 'S2_Time_labels'   : [  'S2 time (mus)'],
                 'nPMT'             : 12}

    test_bins = {'S1_Energy': [   0,     5,   10            ],
                 'S1_Width' : [   0,     2,    4,    6      ],
                 'S1_Time'  : [   0, 1.250,  2.5, 3.75,   5.],
                 'S2_Time'  : [   0, 1.250,  2.5, 3.75,   5.],
                 'S2_Energy': [   0, 7.5e3, 15e3            ],
                 'S2_Height': [   0, 7.5e3, 15e3            ],
                 'S2_XSiPM' : [-100,   -50,    0,   50, 100.],
                 'S2_YSiPM' : [-100,   -50,    0,   50, 100.],
                 'S2_Time'  : [   5,  6.25,  7.5, 8.75,  10.]}

    out_bins, out_labels = monf.pmap_bins(test_dict)

    list_var = ['S1_Energy', 'S1_Width', 'S1_Time', 'S2_Time', 'S2_Energy', 'S2_Height']

    for var_name in list_var:
        assert np.allclose(out_bins[var_name], test_bins[var_name])
        assert test_dict[var_name + '_labels'] == out_labels[var_name]

    assert np.allclose(out_bins['S1_Energy_S1_Width'] [0], test_bins['S1_Energy'])
    assert np.allclose(out_bins['S1_Energy_S1_Width'] [1], test_bins['S1_Width' ])

    assert np.allclose(out_bins['S1_Time_S1_Energy']  [0], test_bins['S1_Time'  ])
    assert np.allclose(out_bins['S1_Time_S1_Energy']  [1], test_bins['S1_Energy'])

    assert np.allclose(out_bins['S2_Time_S2_Energy']  [0], test_bins['S2_Time'  ])
    assert np.allclose(out_bins['S2_Time_S2_Energy']  [1], test_bins['S2_Energy'])

    assert np.allclose(out_bins['S2_Energy_S1_Energy'][0], test_bins['S2_Energy'])
    assert np.allclose(out_bins['S2_Energy_S1_Energy'][1], test_bins['S1_Energy'])

    assert np.allclose(out_bins['S2_XYSiPM']          [0], test_bins['S2_XSiPM' ])
    assert np.allclose(out_bins['S2_XYSiPM']          [1], test_bins['S2_YSiPM' ])

    assert out_labels['S1_Energy_S1_Width'] [0] == test_dict['S1_Energy_labels'][0]
    assert out_labels['S1_Energy_S1_Width'] [1] == test_dict['S1_Width_labels' ][0]

    assert out_labels['S1_Time_S1_Energy']  [0] == test_dict['S1_Time_labels'  ][0]
    assert out_labels['S1_Time_S1_Energy']  [1] == test_dict['S1_Energy_labels'][0]

    assert out_labels['S2_Time_S2_Energy']  [0] == test_dict['S2_Time_labels'  ][0]
    assert out_labels['S2_Time_S2_Energy']  [1] == test_dict['S2_Energy_labels'][0]

    assert out_labels['S2_Energy_S1_Energy'][0] == test_dict['S2_Energy_labels'][0]
    assert out_labels['S2_Energy_S1_Energy'][1] == test_dict['S1_Energy_labels'][0]

    assert out_labels['S2_Energy_S2_Height'][0] == test_dict['S2_Energy_labels'][0]
    assert out_labels['S2_Energy_S2_Height'][1] == test_dict['S2_Height_labels'][0]

    assert out_labels['S2_XYSiPM']          [0] == test_dict['S2_XSiPM_labels' ][0]
    assert out_labels['S2_XYSiPM']          [1] == test_dict['S2_YSiPM_labels' ][0]

    variable_names = ['S1_Energy', 'S1_Width', 'S1_Time', 'S2_Energy', 'S2_Time',
                     'S2_Height', 'S1_Energy_S1_Width', 'S1_Time_S1_Energy',
                     'S2_Time_S2_Energy', 'S2_Energy_S1_Energy', 'S2_Energy_S2_Height',
                     'S2_XYSiPM']

    for i in range(test_dict['nPMT']):
        energy_name = f'PMT{i}_S2_Energy'
        height_name = f'PMT{i}_S2_Height'
        time_name   = f'PMT{i}_S2_Time'
        variable_names.append(energy_name)
        variable_names.append(height_name)
        variable_names.append(  time_name)
        assert out_labels[energy_name] == [f'PMT{i} ' + test_dict['S2_Energy_labels'][0]]
        assert out_labels[height_name] == [f'PMT{i} ' + test_dict['S2_Height_labels'][0]]
        assert out_labels[  time_name] == [f'PMT{i} ' + test_dict['S2_Time_labels'  ][0]]

    for k in out_bins:
        assert k in variable_names

def test_fill_pmap_histos(ICDATADIR):
    test_config_dict = {'S1_Number_bins'   : [-0.50,   10.50,   11],
                        'S1_Width_bins'    : [-0.01,    0.99,   40],
                        'S1_Height_bins'   : [    0,      10,   10],
                        'S1_Energy_bins'   : [    0,      50,   50],
                        'S1_Charge_bins'   : [    0,       2,   20],
                        'S1_Time_bins'     : [    0,     650,  650],

                        'S2_Number_bins'   : [-0.50,   10.50,   11],
                        'S2_Width_bins'    : [    0,      50,   50],
                        'S2_Height_bins'   : [    0,    8000,  100],
                        'S2_Energy_bins'   : [    0,    20e3,  100],
                        'S2_Charge_bins'   : [    0,    3500,  100],
                        'S2_Time_bins'     : [  640,    1300,  660],

                        'S2_NSiPM_bins'    : [-0.50,  500.50,  501],
                        'S2_IdSiPM_bins'   : [-0.50, 1792.50, 1793],
                        'S2_QSiPM_bins'    : [    0,     100,  100],
                        'S2_XSiPM_bins'    : [ -200,     200,   40],
                        'S2_YSiPM_bins'    : [ -200,     200,   40],

                        'S1_Number_labels' : [       "S1 number (#)"],
                        'S1_Width_labels'  : [  r"S1 width ($\mu$s)"],
                        'S1_Height_labels' : [     "S1 height (pes)"],
                        'S1_Energy_labels' : [     "S1 energy (pes)"],
                        'S1_Charge_labels' : [     "S1 charge (pes)"],
                        'S1_Time_labels'   : [   r"S1 time ($\mu$s)"],

                        'S2_Number_labels' : [       "S2 number (#)"],
                        'S2_Width_labels'  : [  r"S2 width ($\mu$s)"],
                        'S2_Height_labels' : [     "S2 height (pes)"],
                        'S2_Energy_labels' : [     "S2 energy (pes)"],
                        'S2_Charge_labels' : [     "S2 charge (pes)"],
                        'S2_Time_labels'   : [   r"S2 time ($\mu$s)"],

                        'S2_NSiPM_labels'  : [     'SiPM number (#)'],
                        'S2_IdSiPM_labels' : [             'SiPM id'],
                        'S2_QSiPM_labels'  : [   'SiPM charge (pes)'],
                        'S2_XSiPM_labels'  : [              'X (mm)'],
                        'S2_YSiPM_labels'  : [              'Y (mm)'],

                        'nPMT'             : 12}

    test_infile = "Kr_pmaps_run4628.h5"
    test_infile = os.path.join(ICDATADIR, test_infile)

    run_number = 4628

    test_histo = monf.fill_pmap_histos(test_infile, run_number, test_config_dict)

    test_checkfile = "Kr_pmaps_histos_run4628.h5"
    test_checkfile = os.path.join(ICDATADIR, test_checkfile)
    check_histo    = histf.get_histograms_from_file(test_checkfile)

    assert set(check_histo.histos) ==  set(test_histo.histos)

    for k, v in check_histo.histos.items():
        assert np.allclose(v.data     , test_histo.histos[k].data     )
        assert np.allclose(v.out_range, test_histo.histos[k].out_range)
        assert np.allclose(v.errors   , test_histo.histos[k].errors   )
        assert             v.title   == test_histo.histos[k].title
        assert             v.labels  == test_histo.histos[k].labels
        for i, bins in enumerate(v.bins):
            assert np.allclose(bins,    test_histo.histos[k].bins  [i])


def test_fill_rwf_var():
    var_dict = defaultdict(list)
    pmt_waveforms  = np.random.uniform(0, 10, size=(  12, 10000))
    sipm_waveforms = np.random.uniform(0, 10, size=(1792, 10000))
    monf.fill_rwf_var(pmt_waveforms , var_dict, "PMT" )
    monf.fill_rwf_var(sipm_waveforms, var_dict, "SiPM")

    assert np.allclose(var_dict['PMT_Baseline']    , np.mean(pmt_waveforms , axis=1))
    assert np.allclose(var_dict['PMT_BaselineRMS'] , np.std (pmt_waveforms , axis=1))
    assert np.allclose(var_dict['SiPM_Baseline']   , np.mean(sipm_waveforms, axis=1))
    assert np.allclose(var_dict['SiPM_BaselineRMS'], np.std (sipm_waveforms, axis=1))


def test_rwf_bins():
    test_dict = {'PMT_Baseline_bins'       : [2300, 2700, 400],
                 'PMT_BaselineRMS_bins'    : [   0,   10, 100],
                 'PMT_nSensors_bins'       : [-0.5,   12, 100],
                 'SiPM_Baseline_bins'      : [   0,  100, 100],
                 'SiPM_BaselineRMS_bins'   : [   0,   10, 100],
                 'SiPM_nSensors_bins'      : [-0.5,   12, 100],


                 'PMT_Baseline_labels'     : ["ADCs"]           ,
                 'PMT_BaselineRMS_labels'  : ["ADCs"]           ,
                 'PMT_nSensors_labels'     : ["Number of PMTs"] ,
                 'SiPM_Baseline_labels'    : ["ADCs"]           ,
                 'SiPM_BaselineRMS_labels' : ["ADCs"]           ,
                 'SiPM_nSensors_labels'    : ["Number of SiPMs"],

                 'n_baseline'              : 10000 }

    out_bins, out_labels, out_baseline = monf.rwf_bins(test_dict)

    bins = test_dict['PMT_Baseline_bins']
    assert np.allclose(out_bins['PMT_Baseline']    , [np.linspace(bins[0], bins[1], bins[2] + 1)])
    bins = test_dict['PMT_BaselineRMS_bins']
    assert np.allclose(out_bins['PMT_BaselineRMS'] , [np.linspace(bins[0], bins[1], bins[2] + 1)])
    bins = test_dict['PMT_nSensors_bins']
    assert np.allclose(out_bins['PMT_nSensors']    , [np.linspace(bins[0], bins[1], bins[2] + 1)])
    bins = test_dict['SiPM_Baseline_bins']
    assert np.allclose(out_bins['SiPM_Baseline']   , [np.linspace(bins[0], bins[1], bins[2] + 1)])
    bins = test_dict['SiPM_BaselineRMS_bins']
    assert np.allclose(out_bins['SiPM_BaselineRMS'], [np.linspace(bins[0], bins[1], bins[2] + 1)])
    bins = test_dict['SiPM_nSensors_bins']
    assert np.allclose(out_bins['SiPM_nSensors']   , [np.linspace(bins[0], bins[1], bins[2] + 1)])

    assert out_labels['PMT_Baseline']    [0] == test_dict['PMT_Baseline_labels']    [0]
    assert out_labels['PMT_BaselineRMS'] [0] == test_dict['PMT_BaselineRMS_labels'] [0]
    assert out_labels['PMT_nSensors']    [0] == test_dict['PMT_nSensors_labels']    [0]
    assert out_labels['SiPM_Baseline']   [0] == test_dict['SiPM_Baseline_labels']   [0]
    assert out_labels['SiPM_BaselineRMS'][0] == test_dict['SiPM_BaselineRMS_labels'][0]
    assert out_labels['SiPM_nSensors']   [0] == test_dict['SiPM_nSensors_labels']   [0]
    assert out_baseline                      == test_dict['n_baseline']


def test_fill_rwf_histos(ICDATADIR):
    test_config_dict = {'PMT_Baseline_bins'       : [  2300,   2700,  400],
                        'PMT_BaselineRMS_bins'    : [     0,     10,  100],
                        'PMT_nSensors_bins'       : [  -0.5,   12.5,   13],
                        'SiPM_Baseline_bins'      : [     0,    100,  100],
                        'SiPM_BaselineRMS_bins'   : [     0,     10,  100],
                        'SiPM_nSensors_bins'      : [1750.5, 1800.5,   50],

                        'PMT_Baseline_labels'     : ["PMT Baseline (ADC)"],
                        'PMT_BaselineRMS_labels'  : ["PMT Baseline RMS (ADC)"],
                        'PMT_nSensors_labels'     : ["Number of PMTs"],
                        'SiPM_Baseline_labels'    : ["SiPM Baseline (ADC)"],
                        'SiPM_BaselineRMS_labels' : ["SiPM Baseline RMS (ADC)"],
                        'SiPM_nSensors_labels'    : ["Number of SiPMs"],

                        'n_baseline'              : 48000}

    test_infile = "irene_bug_Kr_ACTIVE_7bar_RWF.h5"
    test_infile = os.path.join(ICDATADIR, test_infile)

    test_histo = monf.fill_rwf_histos(test_infile, test_config_dict)

    test_checkfile = "irene_bug_Kr_ACTIVE_7bar_RWF_histos.h5"
    test_checkfile = os.path.join(ICDATADIR, test_checkfile)
    check_histo    = histf.get_histograms_from_file(test_checkfile)

    assert set(check_histo.histos) ==  set(test_histo.histos)

    for k, v in check_histo.histos.items():
        assert np.allclose(v.data     , test_histo.histos[k].data     )
        assert np.allclose(v.out_range, test_histo.histos[k].out_range)
        assert np.allclose(v.errors   , test_histo.histos[k].errors   )
        assert             v.title   == test_histo.histos[k].title
        assert             v.labels  == test_histo.histos[k].labels
        for i, bins in enumerate(v.bins):
            assert np.allclose(bins,    test_histo.histos[k].bins[i])
