import os

from pandas import DataFrame

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
                     nmax        = nrequired))


    dorothea = Dorothea(**conf)

    dorothea.run()
    cnt  = dorothea.end()
    nevt_in  = cnt.n_events_tot
    nevt_out = cnt.nevt_out
    if nrequired > 0:
        assert nrequired    == nevt_in
        assert nevt_out     <= nevt_in

    dst = load_dst(PATH_OUT, "DST", "Events")
    assert len(set(dst.event)) == nevt_out

    df = DataFrame.from_dict(dict(
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
