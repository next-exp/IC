import numpy as np

from   invisible_cities.database import load_db
from   invisible_cities.core.system_of_units_c import SystemOfUnits

units = SystemOfUnits()


class City:

    def __init__(self, run_number=0, files_in = None):

        self.run_number = run_number
        self.files_in   = files_in

        DataPMT  = load_db.DataPMT (run_number)
        DataSiPM = load_db.DataSiPM(run_number)

        # This is JCK-1: text reveals symmetry!
        self.xs              = DataSiPM.X.values
        self.ys              = DataSiPM.Y.values
        self.adc_to_pes      = abs(DataPMT.adc_to_pes.values).astype(np.double)
        self.sipm_adc_to_pes = DataSiPM.adc_to_pes.values    .astype(np.double)
        self.coeff_c         = DataPMT.coeff_c.values        .astype(np.double)
        self.coeff_blr       = DataPMT.coeff_blr.values      .astype(np.double)

        # default parameters
        self.nprint       = 1000000
        self.signal_start =       0 # microseconds
        self.signal_end   =    1200 # microseconds

        self.input_files = files_in

    def set_print(self, nprint=10):
        """Print frequency."""
        self.nprint = nprint

    def set_input_files(self, input_files):
        """Set the input files."""
        self.input_files = input_files

class DeconvolutionCity(City):

    def __init__(self,
                 run_number  = 0,
                 files_in    = None,
                 n_baseline  = 28000,
                 thr_trigger = 5 * units.adc,
                 n_MAU       =   100,
                 thr_MAU     = 3 * units.adc):

        City.__init__(self, run_number=run_number, files_in=files_in)

        # BLR parameters
        self.n_baseline  = n_baseline
        self.thr_trigger = thr_trigger

        # Parameters of the MAU used to remove low frequency noise.
        self.  n_MAU =   n_MAU
        self.thr_MAU = thr_MAU

    def set_BLR(self, n_baseline, thr_trigger):
        self.n_baseline  = n_baseline
        self.thr_trigger = thr_trigger

    def set_MAU(self, n_MAU, thr_MAU):
        self.  n_MAU =   n_MAU
        self.thr_MAU = thr_MAU


