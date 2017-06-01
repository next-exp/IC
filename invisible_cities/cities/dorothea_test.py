import os

from pytest import mark
from pytest import fixture
from pandas import DataFrame

from . dorothea import Dorothea
from . dorothea import DOROTHEA

from .. reco.dst_functions import load_dst
from .. core.test_utils    import assert_dataframes_close
from .. core               import system_of_units as units

def test_dorothea_KrMC(config_tmpdir, KrMC_pmaps):
    # NB: avoid taking defaults for PATH_IN and PATH_OUT
    # since they are in general test-specific
    # NB: avoid taking defaults for run number (test-specific)

    PATH_IN =  KrMC_pmaps[0]
    PATH_OUT = os.path.join(str(config_tmpdir), 'KrDST.h5')

    dorothea = Dorothea(run_number = 0,
                        files_in   = [PATH_IN],
                        file_out   = PATH_OUT,

                        drift_v    = 1 * units.mm/units.mus,
                        S1_Emin     =      0,
                        S1_Emax     =     30,
                        S1_Lmin     =      4,
                        S1_Lmax     =     20,
                        S1_Hmin     =    0.5,
                        S1_Hmax     =     10,
                        S1_Ethr     =    0.5,

                        S2_Nmax     =      1,
                        S2_Emin     =    1e3,
                        S2_Emax     =    1e8,
                        S2_Lmin     =      1,
                        S2_Lmax     =     20,
                        S2_Hmin     =    500,
                        S2_Hmax     =    1e5,
                        S2_Ethr     =      1,
                        S2_NSIPMmin =      2,
                        S2_NSIPMmax =   1000)

    nrequired = 10
    nevt_in, nevt_out, in_out_ratio = dorothea.run(max_evt=nrequired)
    if nrequired > 0:
        assert nrequired    == nevt_in
        assert nevt_out     <= nevt_in
        assert in_out_ratio >= 1

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

    assert_dataframes_close(dst, df, False, rtol=1e-6)
