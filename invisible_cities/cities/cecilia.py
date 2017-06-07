"""
code: cecilia.py
description: simulation of trigger for the NEXT detector.
author: P. Ferrario
IC core team: Jacek Generowicz, JJGC, G. Martinez, J.A. Hernando, J.M Benlloch
package: invisible cities. See release notes and licence
last changed: 01-02-2017
"""

import sys
import textwrap

from glob      import glob
from time      import time
from functools import partial
from argparse  import Namespace

import numpy as np
import tables as tb

from .. core.configure          import configure
from .. core.configure          import print_configuration
from .. core.system_of_units_c  import units
from .. core.sensor_functions   import convert_channel_id_to_IC_id
from .. reco                    import tbl_functions    as tbl
from .. reco                    import peak_functions_c as cpf
from .. database                import load_db          as db
from .. reco.nh5                import RunInfo
from .. reco.nh5                import EventInfo
from .. reco.params             import minmax
from .. reco.params             import PeakData
from .. reco.params             import TriggerParams
from .  base_cities             import DeconvolutionCity
from .. io.mc_io                import mc_track_writer
from .. io.run_and_event_io     import run_and_event_writer
from .. io.rwf_io               import rwf_writer
from .. io.fee_io               import write_FEE_table
from .. filters.trigger_filters import TriggerFilter

class Cecilia(DeconvolutionCity):
    "The city of CECILIA simulates the trigger."
    def __init__(self,
                 run_number            = 0,
                 files_in              = None,
                 file_out              = None,
                 compression           = 'ZLIB4',
                 nprint                = 10000,
                 n_baseline            = 28000,
                 thr_trigger           =     5 * units.adc,
                 acum_discharge_length =  5000,
                 # Parameters added at this level
                 trigger_channels      = tuple(range(12)),
                 min_number_channels   =     5,
                 height = minmax(min   =    15,
                                 max   =  1000),
                 charge = minmax(min   =  3000,
                                 max   = 20000),
                 width  = minmax(min   =  4000,
                                 max   = 12000),
                 data_MC_ratio         =     0.8):

        super().__init__(run_number            = run_number,
                         files_in              = files_in,
                         file_out              = file_out,
                         compression           = compression,
                         nprint                = nprint,
                         n_baseline            = n_baseline,
                         thr_trigger           = thr_trigger,
                         acum_discharge_length = acum_discharge_length)

        self.trigger_params = TriggerParams(
            trigger_channels    = trigger_channels,
            min_number_channels = min_number_channels,
            charge              = charge * data_MC_ratio,
            height              = height * data_MC_ratio,
            width               = width)

        self.event_in = 0
        self.event_out = 0

    def run(self, nmax):
        self.display_IO_info(nmax)
        sp = sensor_params = self.get_sensor_params(self.input_files[0])
        self.print_configuration(sensor_params)

        # loop over input files
        with tb.open_file(self.output_file, "w",
                          filters = tbl.filters(self.compression)) as h5out:
            # Create writers
            RWF = partial(rwf_writer,  h5out,   group_name='RD')
            writers = Namespace(
                run_and_event = run_and_event_writer(h5out),
                mc            =      mc_track_writer(h5out) if self.monte_carlo else None,
                # 3 variations on the  RWF writer theme
                rwf  = RWF(table_name='pmtrwf' , n_sensors=sp.NPMT , waveform_length=sp.PMTWL),
                cwf  = RWF(table_name='pmtblr' , n_sensors=sp.NPMT , waveform_length=sp.PMTWL),
                sipm = RWF(table_name='sipmrwf', n_sensors=sp.NSIPM, waveform_length=sp.SIPMWL),
            )

            # Create and store Front-End Electronics parameters (for the PMTs)
            write_FEE_table(h5out)

            nevt_in, nevt_out = self._file_loop(writers, nmax)
        print(textwrap.dedent("""
                              Number of events in : {}
                              Number of events out: {}
                              Ratio               : {}
                              """.format(nevt_in, nevt_out, nevt_out / nevt_in)))
        return nevt_in, nevt_out

    def _file_loop(self, writers, nmax):
        nevt_in = nevt_out = 0
        trigger_filter = TriggerFilter(self.trigger_params)

        for filename in self.input_files:
            print("Opening file {filename}".format(**locals()), end='... ')

            with tb.open_file(filename, "r") as h5in:
                NEVT, pmtrwf, sipmrwf, pmtblr = self.get_rwf_vectors(h5in)
                events_info = self.get_run_and_event_info(h5in)

                nevt_in, nevt_out, max_events_reached = self._event_loop(
                    NEVT, pmtrwf, sipmrwf, pmtblr, events_info,
                    writers,
                    nmax, nevt_in, nevt_out, h5in, trigger_filter)

            if max_events_reached:
                print('Max events reached')
                break
            else:
                print("OK")

        return nevt_in, nevt_out

    def _event_loop(self, NEVT, pmtrwf, sipmrwf, pmtblr, events_info,
                    write,
                    nmax, nevt_in, nevt_out, h5in, trigger_pass):
        max_events_reached = False
        for evt in range(NEVT):
            nevt_in += 1
            if self.max_events_reached(nmax, nevt_in):
                max_events_reached = True
                break
            event_number, timestamp = self.event_and_timestamp(evt, events_info)
            peak_data = self._emulate_trigger(pmtrwf[evt])

            if not trigger_pass(peak_data):
                continue
            nevt_out += 1

            write.mc(h5in.root.MC.MCTracks, event_number)
            write.run_and_event(self.run_number, event_number, timestamp)
            write.rwf (pmtrwf [evt])
            write.cwf (pmtblr [evt])
            write.sipm(sipmrwf[evt])

            self.conditional_print(evt, nevt_in)
        return nevt_in, nevt_out, max_events_reached

    def _emulate_trigger(self, RWF):
        # Emulate deconvolution in the FPGA
        CWF = self.deconv_pmt(RWF)
        # Emulate FPGA
        return self._peak_computation_in_FPGA(CWF, self.trigger_params.trigger_channels)

    def _peak_computation_in_FPGA(self, CWF, channel_id_selection):
        # Translate electroninc channel id to IC index
        IC_ids_selection = convert_channel_id_to_IC_id(self.DataPMT, channel_id_selection)
        # Emulate zero suppression in the FPGA
        wfzs = [cpf.wfzs(CWF[pmt_id].astype(np.double),
                         threshold = self.trigger_params.height.min)
                for pmt_id in IC_ids_selection]
        # Emulate peak search in the FPGA
        s12s =  [cpf.find_S12(content, index) for content, index in wfzs]

        peak_data = {}
        for s12, channel_no in zip(s12s, IC_pmt_ids_selection):
            pd = {}
            for peak_no, peak in s12.items():
                pd[peak_no] = PeakData(width  = peak.t[-1] - peak.t[0],
                                       charge = np.sum(peak.E),
                                       height = np.max(peak.E))
            peak_data[channel_no] = pd

        return peak_data


def CECILIA(argv=sys.argv):
    CFP = configure(argv)
    files_in = glob(CFP.FILE_IN)
    files_in.sort()
    cecilia = Cecilia(files_in            = files_in,
                      file_out            = CFP.FILE_OUT,
                      trigger_channels    = CFP.TR_CHANNELS,
                      min_number_channels = CFP.MIN_NUMBER_CHANNELS,
                      height = minmax(min = CFP.MIN_HEIGHT,
                                      max = CFP.MAX_HEIGHT),
                      charge = minmax(min = CFP.MIN_CHARGE,
                                      max = CFP.MAX_CHARGE),
                      width  = minmax(min = CFP.MIN_WIDTH,
                                      max = CFP.MAX_WIDTH),
                      data_MC_ratio       = CFP.DATA_MC_RATIO
    )

    # TODO
    # trigger_type = CFP.TRIGGER


    nevts = CFP.NEVENTS if not CFP.RUN_ALL else -1
    t0 = time()
    nevt_in, nevt_out = cecilia.run(nmax=nevts)
    t1 = time()
    dt = t1 - t0

    print("run {} evts in {} s, time/event = {}".format(nevt_in, dt, dt/nevt_in))


if __name__ == "__main__":
    CECILIA(sys.argv)
