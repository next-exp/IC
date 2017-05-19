import os

from pytest import mark
from pytest import fixture
from pandas import DataFrame

from . dorothea import Dorothea
from . dorothea import DOROTHEA

from .. reco.dst_functions import load_dst
from .. core.test_utils    import assert_dataframes_close

@fixture(scope='module')
def conf_file_name_mc(config_tmpdir):
    # Specifies a name for a MC configuration file. Also, default number
    # of events set to 1.

    conf_file_name = str(config_tmpdir.join('dorothea_mc.conf'))
    Dorothea.write_config_file(conf_file_name,
                               PATH_OUT = str(config_tmpdir),
                               NEVENTS  = 1)
    return conf_file_name


def test_command_line_dorothea_KrMC(conf_file_name_mc, config_tmpdir, KrMC_pmaps):
    # NB: avoid taking defaults for PATH_IN and PATH_OUT
    # since they are in general test-specific
    # NB: avoid taking defaults for run number (test-specific)

    PATH_IN =  KrMC_pmaps[0]
    PATH_OUT = os.path.join(str(config_tmpdir), 'KrDST.h5')

    nrequired, nactual, nout = DOROTHEA(['DOROTHEA',
                                         '-c', conf_file_name_mc,
                                         '-i', PATH_IN,
                                         '-o', PATH_OUT,
                                         '-n', '10',
                                         '-r', '0'])
    if nrequired > 0:
        assert nrequired == nactual

    dst = load_dst(PATH_OUT, "DST", "Events")
    assert len(set(dst.event)) == nout

    df = DataFrame.from_dict(dict(
            event = [31],
            time  = [0.031],
            peak  = [0],
            nS2   = [1],
            S1w   = [125.0],
            S1h   = [1.423625],
            S1e   = [5.06363],
            S1t   = [100125.0],
            S2e   = [5375.89229202],
            S2w   = [8.97875],
            S2h   = [1049.919067],
            S2q   = [356.082108974],
            S2t   = [453937.5],
            Nsipm = [6],
            DT    = [353.8125],
            Z     = [353.8125],
            X     = [-125.205608],
            Y     = [148.305353],
            R     = [194.089984],
            Phi   = [2.271938],
            Xrms  = [6.762344],
            Yrms  = [4.710678]))

    assert_dataframes_close(dst, df, False, rtol=1e-6)
