import os

from pytest import mark

from .. core.configure       import configure
from .. core.system_of_units import ns

from .  zaira import zaira

@mark.slow
def test_zaira_KrMC(config_tmpdir, ICDIR):
    # NB: avoid taking defaults for PATH_IN and PATH_OUT
    # since they are in general test-specific
    # NB: avoid taking defaults for run number (test-specific)

    PATH_IN =  os.path.join(ICDIR, 'database/test_data', 'KrDST_MC.h5')
    PATH_OUT = os.path.join(config_tmpdir,               'KrCorr.h5')


    conf = configure('dummy invisible_cities/config/zaira.conf'.split())
    conf.update(dict(files_in   = PATH_IN,
                     file_out   = PATH_OUT,
                     lifetime   = 1e6 * ns,
                     xbins      = 20,
                     ybins      = 20))

    cnt = zaira(**conf)

    assert cnt.events_in  >  0
    assert cnt.events_out <= cnt.events_in
