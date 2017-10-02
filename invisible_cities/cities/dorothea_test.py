import os

import numpy  as np
import pandas as pd

from . dorothea import Dorothea

from .. io.dst_io import load_dst
from .. core.testing_utils    import assert_dataframes_close
from .. core.configure     import configure
from .. core.system_of_units import pes, mm, mus, ns

def test_dorothea_KrMC(config_tmpdir, KrMC_pmaps):
    # NB: avoid taking defaults for PATH_IN and PATH_OUT
    # since they are in general test-specific
    # NB: avoid taking defaults for run number (test-specific)

    PATH_IN =  KrMC_pmaps[0]

    PATH_OUT = os.path.join(config_tmpdir, 'KrDST.h5')
    nrequired = 10
    conf = configure('dummy invisible_cities/config/dorothea.conf'.split())
    conf.update(dict(run_number = 0,
                     files_in   = PATH_IN,
                     file_out   = PATH_OUT,

                     drift_v     =      1 * mm / mus,
                     s1_nmin     =      1,
                     s1_nmax     =      1,
                     s1_emin     =      0 * pes,
                     s1_emax     =     30,
                     s1_wmin     =    100 * ns,
                     s1_wmax     =    500 * ns,
                     s1_hmin     =    0.5 * pes,
                     s1_hmax     =     10 * pes,
                     s1_ethr     =    0.37 * pes,
                     s2_nmin     =      1,
                     s2_nmax     =      2,
                     s2_emin     =    1e3 * pes,
                     s2_emax     =    1e8 * pes,
                     s2_wmin     =      1 * mus,
                     s2_wmax     =     20 * mus,
                     s2_hmin     =    500 * pes,
                     s2_hmax     =    1e5 * pes,
                     s2_ethr     =      1 * pes,
                     s2_nsipmmin =      2,
                     s2_nsipmmax =   1000,
                     event_range = (0, nrequired)))


    dorothea = Dorothea(**conf)

    dorothea.run()
    cnt  = dorothea.end()
    nevt_in  = cnt.n_events_tot
    nevt_out = cnt.n_events_selected
    if nrequired > 0:
        assert nrequired    == nevt_in
        assert nevt_out     <= nevt_in

    dst = load_dst(PATH_OUT, "DST", "Events")
    assert len(set(dst.event)) == nevt_out

    df = pd.DataFrame.from_dict(dict(
            event = [    31          ],
            time  = [     0.031      ],
            peak  = [     0          ],
            nS2   = [     1          ],
            S1w   = [   125          ],
            S1h   = [     1.423625   ],
            S1e   = [     5.06363    ],
            S1t   = [100125.0        ],
            S2e   = [  5375.89229202 ],
            S2w   = [     8.97875    ],
            S2h   = [  1049.919067   ],
            S2q   = [   356.082108974],
            S2t   = [453937.5        ],
            Nsipm = [     6          ],
            DT    = [   353.8125     ],
            Z     = [   353.8125     ],
            X     = [  -125.205608   ],
            Y     = [   148.305353   ],
            R     = [   194.089984   ],
            Phi   = [     2.271938   ],
            Xrms  = [     6.762344   ],
            Yrms  = [     4.710678   ]))

    assert_dataframes_close(dst, df, False, rtol=1e-2)


def test_dorothea_filter_events(config_tmpdir, Kr_pmaps_run4628):
    PATH_IN =  Kr_pmaps_run4628

    PATH_OUT = os.path.join(config_tmpdir, 'KrDST_4628.h5')
    nrequired = 50
    conf = configure('dummy invisible_cities/config/dorothea.conf'.split())
    conf.update(dict(run_number = 4628,
                     files_in   = PATH_IN,
                     file_out   = PATH_OUT,

                     drift_v     =      2 * mm / mus,
                     s1_nmin     =      1,
                     s1_nmax     =      1,
                     s1_emin     =      1 * pes,
                     s1_emax     =     30 * pes,
                     s1_wmin     =    100 * ns,
                     s1_wmax     =    300 * ns,
                     s1_hmin     =      1 * pes,
                     s1_hmax     =      5 * pes,
                     s1_ethr     =    0.5 * pes,
                     s2_nmin     =      1,
                     s2_nmax     =      2,
                     s2_emin     =    1e3 * pes,
                     s2_emax     =    1e4 * pes,
                     s2_wmin     =      2 * mus,
                     s2_wmax     =     20 * mus,
                     s2_hmin     =    1e3 * pes,
                     s2_hmax     =    1e5 * pes,
                     s2_ethr     =      1 * pes,
                     s2_nsipmmin =      5,
                     s2_nsipmmax =     30,
                     event_range = (0, nrequired)))

    events_pass = [ 1,  4, 10, 19, 20, 21, 26,
                   26, 29, 33, 41, 43, 45, 46]
    peak_pass   = [ 0,  0,  0,  0,  0,  0,  0,
                    1,  0,  0,  0,  0,  0,  0]

    dorothea = Dorothea(**conf)

    dorothea.run()
    cnt  = dorothea.end()
    nevt_in  = cnt.n_events_tot
    nevt_out = cnt.n_events_selected
    assert nrequired    == nevt_in
    assert nevt_out     == len(set(events_pass))

    dst = load_dst(PATH_OUT, "DST", "Events")
    assert len(set(dst.event.values)) == nevt_out

    assert np.all(dst.event.values == events_pass)
    assert np.all(dst.peak.values  ==   peak_pass)

def test_dorothea_issue_347(KrMC_pmaps, config_tmpdir):
    PATH_IN =  KrMC_pmaps[0]
    PATH_OUT = os.path.join(config_tmpdir, 'KrDST.h5')
    conf = configure('dummy invisible_cities/config/dorothea_with_corona.conf'.split())
    # with this parameters Corona will find several clusters
    conf.update(dict(files_in   = PATH_IN,
                     file_out   = PATH_OUT,
                     lm_radius     = 0.1,
                     new_lm_radius = 0.2,
                     msipm         = 1))
    dorothea = Dorothea(**conf)
    dorothea.run()
    dorothea.end()
