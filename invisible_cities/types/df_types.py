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
summary_type : dict
    Dtype schema for the per-event tracking summary table.
tracks_type : dict
    Dtype schema for the per-track reconstruction table.
"""

import numpy as np

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

summary_type = dict( event         = np.int64
                   , evt_energy    = np.float64
                   , evt_charge    = np.float64
                   , evt_ntrks     = int
                   , evt_nhits     = int
                   , evt_x_avg     = np.float64
                   , evt_y_avg     = np.float64
                   , evt_z_avg     = np.float64
                   , evt_r_avg     = np.float64
                   , evt_x_min     = np.float64
                   , evt_y_min     = np.float64
                   , evt_z_min     = np.float64
                   , evt_r_min     = np.float64
                   , evt_x_max     = np.float64
                   , evt_y_max     = np.float64
                   , evt_z_max     = np.float64
                   , evt_r_max     = np.float64
                   , evt_out_of_map = bool
                   )


tracks_type = dict( event            = np.int64
                  , trackID          = int
                  , energy           = np.float64
                  , length           = np.float64
                  , numb_of_voxels   = int
                  , numb_of_hits     = int
                  , numb_of_tracks   = int
                  , x_min            = np.float64
                  , y_min            = np.float64
                  , z_min            = np.float64
                  , r_min            = np.float64
                  , x_max            = np.float64
                  , y_max            = np.float64
                  , z_max            = np.float64
                  , r_max            = np.float64
                  , x_ave            = np.float64
                  , y_ave            = np.float64
                  , z_ave            = np.float64
                  , r_ave            = np.float64
                  , extreme1_x       = np.float64
                  , extreme1_y       = np.float64
                  , extreme1_z       = np.float64
                  , extreme2_x       = np.float64
                  , extreme2_y       = np.float64
                  , extreme2_z       = np.float64
                  , blob1_x          = np.float64
                  , blob1_y          = np.float64
                  , blob1_z          = np.float64
                  , blob2_x          = np.float64
                  , blob2_y          = np.float64
                  , blob2_z          = np.float64
                  , eblob1           = np.float64
                  , eblob2           = np.float64
                  , ovlp_blob_energy = np.float64
                  , vox_size_x       = np.float64
                  , vox_size_y       = np.float64
                  , vox_size_z       = np.float64
                  )
