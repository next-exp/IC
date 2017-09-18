import os
import numpy  as np

from .. core.system_of_units import *
from .. core.testing_utils   import assert_dataframes_close
from .  penthesilea          import Penthesilea
from .. core.configure       import configure
from .. io                   import dst_io as dio


def test_penthesilea_run_on_Kr_MC_ipmt_pmaps_5evt(ICDIR, config_tmpdir):
    #PATH_IN  = os.path.join(ICDIR, 'database/test_data/Kr_MC_ipmt_pmaps_5evt.h5')
    PATH_OUT  = os.path.join(config_tmpdir,'KR_MC_penth_6evt.h5')
    PATH_COMP = os.path.join(ICDIR, 'database/test_data/KR_MC_penth_6evt.h5')
    conf = configure('dummy invisible_cities/config/penthesilea.conf'.split())
    conf.update(dict(file_out      = PATH_OUT,
                     event_range   = (10,19),
                     rebin         =  2,
                     drift_v       = 1 * mm / mus, # Expected drift velocity
                     s1_nmin       =    1,
                     s1_nmax       =    10,
                     s1_emin       =     0 * pes, # Min S1 energy integral
                     s1_emax       =  1e+6 * pes, # Max S1 energy integral
                     s1_wmin       =   100 * ns, # min width for S1
                     s1_wmax       =   500 * ns, # Max width
                     s1_hmin       =     0 * pes, # Min S1 height
                     s1_hmax       =  1e+6 * pes, # Max S1 height
                     s1_ethr       =   0.5 * pes, # Energy threshold for S1
                     s2_nmin       =     1,
                     s2_nmax       =     10,      # Max number of S2 signals
                     s2_emin       =     0 * pes, # Min S2 energy integral
                     s2_emax       =  1e+6 * pes, # Max S2 energy integral in pes
                     s2_wmin       =     3 * mus, # Min width
                     s2_wmax       =    10 * ms,  # Max width
                     s2_hmin       =     0 * pes, # Min S2 height
                     s2_hmax       =  1e+6 * pes, # Max S2 height
                     s2_nsipmmin   =     1,       # Min number of SiPMs touched
                     s2_nsipmmax   =   100,       # Max number of SiPMs touched
                     s2_ethr       =   0.5 * pes,  # Energy threshold for S2
                     msipm         =     1,
                     qthr          =     2 * pes,
                     qlm           =     5 * pes,
                     lm_radius     =   0 * mm,
                     new_lm_radius =    15 * mm))

    penthesilea = Penthesilea(**conf)
    penthesilea.run()
    cnt = penthesilea.end()
    assert cnt.n_events_tot == 9
    assert cnt.n_events_selected == 6
    pent = dio.load_dst(PATH_OUT , 'RECO', 'Events')
    comp = dio.load_dst(PATH_COMP, 'RECO', 'Events')
    assert_dataframes_close(pent, comp)
