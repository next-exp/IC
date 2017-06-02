import os

from pytest import mark

from . zaira import Zaira

@mark.slow
def test_zaira_KrMC(config_tmpdir, ICDIR):
    # NB: avoid taking defaults for PATH_IN and PATH_OUT
    # since they are in general test-specific
    # NB: avoid taking defaults for run number (test-specific)

    PATH_IN =  os.path.join(ICDIR,
                            'database/test_data', 'KrDST_MC.h5')
    PATH_OUT = os.path.join(str(config_tmpdir), 'KrCorr.h5')

    zaira = Zaira(run_number = 0,
                  files_in   = [PATH_IN],
                  file_out   = PATH_OUT,
                  lifetime   = 1e6)
    dst_size = zaira.run()

    assert dst_size > 0
