import os
from   os import path

import numpy  as np
import tables as tb

from pytest import mark

from hypothesis             import given
from hypothesis.strategies  import floats
from hypothesis.extra.numpy import arrays

from .. database import load_db

from .        import peak_functions_c as cpf
from .        import tbl_functions    as tbl
from .        import wfm_functions    as wfm
from . params import CalibVectors
from . params import DeconvParams


def ndarrays_of_shape(shape, lo=-1000.0, hi=1000.0):
    return arrays('float64', shape=shape,
                  elements=floats(min_value=lo, max_value=hi))


@mark.slow
def test_compare_cwf_blr():
    """Test functions cwf_from_rwf() and compare_cwf_blr().
    The test:
    1) Computes CWF from RWF (function cwf_from_rwf())
    2) Computes the difference between CWF and BLR (compare_cwf_blr())
    3) Asserts that the differences are small.
    For 10 events and 12 PMTs per event, all differences are less than 0.1 %
    Input file (needed in repository): electrons_40keV_z250_RWF.h5
    """

    RWF_file = path.join(os.environ['ICDIR'],
                         'database/test_data/electrons_40keV_z250_RWF.h5')
    h5rwf = tb.open_file(RWF_file,'r')
    pmtrwf, pmtblr, sipmrwf = tbl.get_vectors(h5rwf)
    NEVT, NPMT, PMTWL = pmtrwf.shape


    deconv = DeconvParams(n_baseline  = 28000,
                          thr_trigger =     5)

    run_number = 0
    DataPMT = load_db.DataPMT(run_number)
    DataSiPM = load_db.DataSiPM(run_number)

    calib = CalibVectors(channel_id = DataPMT.ChannelID.values,
                         coeff_blr = abs(DataPMT.coeff_blr   .values),
                         coeff_c = abs(DataPMT.coeff_c   .values),
                         adc_to_pes = DataPMT.adc_to_pes.values,
                         adc_to_pes_sipm = DataSiPM.adc_to_pes.values,
                         pmt_active = np.nonzero(DataPMT.Active.values)[0].tolist())

    CWF = wfm.cwf_from_rwf(pmtrwf, range(NEVT), calib, deconv)
    diff = wfm.compare_cwf_blr(CWF, pmtblr,
                               event_list=range(NEVT), window_size=200)

    assert max(diff) < 0.1
