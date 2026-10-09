"""
Column dtype schemas for detector dataframes.

Attributes
----------
hit_type : dict
    Dtype schema for hit rows. It covers event and peak identifiers, peak
    positions, hit coordinates, charge, and energy columns.
kr_events_type : dict
    Dtype schema for KDST event rows, including S1/S2 peak summaries, drift
    information, and reconstructed position quantities.
"""

import numpy  as np

hit_type = dict( event = int
               , time  = float
               , npeak = np.uint16
               , Xpeak = float
               , Ypeak = float
               , X     = float
               , Y     = float
               , Z     = float
               , Q     = float
               , E     = float
               , Ec    = float
               )

kr_events_type = dict( event   = int
                     , time    = float
                     , s1_peak = np.uint16
                     , s2_peak = np.uint16
                     , nS1     = np.uint16
                     , nS2     = np.uint16
                     , S1w     = float
                     , S1h     = float
                     , S1e     = float
                     , S1t     = float
                     , S2w     = float
                     , S2h     = float
                     , S2e     = float
                     , S2q     = float
                     , S2t     = float
                     , qmax    = float
                     , Nsipm   = np.uint16
                     , DT      = float
                     , Z       = float
                     , X       = float
                     , Y       = float
                     , R       = float
                     , Phi     = float
                     , Xrms    = float
                     , Yrms    = float
                     )
