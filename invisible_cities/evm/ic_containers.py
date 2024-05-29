"""
code: ic_cotainers.py
description: namedtuples describing miscellaenous containers to
pass info around.

credits: see ic_authors_and_legal.rst in /doc

last revised: JJGC, 10-July-2017
"""

import sys

from collections import namedtuple


this_module = sys.modules[__name__]

def _add_namedtuple_in_this_module(name, attribute_names):
    new_nametuple = namedtuple(name, attribute_names)
    setattr(this_module, name, new_nametuple)

for name, attrs in (
        ('SensorData'     , 'NPMT PMTWL NSIPM SIPMWL '),
        ('DeconvParams'   , 'n_baseline thr_trigger'),
        ('CalibVectors'   , 'channel_id coeff_blr coeff_c adc_to_pes adc_to_pes_sipm pmt_active'),
        ('S12Params'      , 'time stride length rebin_stride'),
        ('ZsWf'           , 'indices energies'),
        ('FitFunction'    , 'fn values errors chi2 pvalue cov infodict mesg ier'),
        ('TriggerParams'  , 'trigger_channels min_number_channels charge height width'),
        ('SensorParams'   , 'spectra peak_range min_bin_peak max_bin_peak half_peak_width p1pe_seed lim_ped'),
        ('PedestalParams' , 'gain gain_min gain_max sigma sigma_min sigma_max')):
    _add_namedtuple_in_this_module(name, attrs)

# Leave nothing but the namedtuple types in the namespace of this module
del name, namedtuple, sys, this_module, _add_namedtuple_in_this_module
