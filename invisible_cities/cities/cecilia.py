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
from .. core.ic_types           import minmax
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
    def __init__(self, **kwds):

        super().__init__(**kwds)

        conf = self.conf

        height = minmax(min = conf.min_height, max = conf.max_height)
        charge = minmax(min = conf.min_charge, max = conf.max_charge)
        width  = minmax(min = conf.min_width , max = conf.max_width )

        self.trigger_params = TriggerParams(
            trigger_channels    = conf.tr_channels,
            min_number_channels = conf.min_number_channels,
            charge              = charge * conf.data_mc_ratio,
            height              = height * conf.data_mc_ratio,
            width               = width)

        self.event_in = 0
        self.event_out = 0

    def run(self):
        self.display_IO_info()
        sp = self.get_sensor_params(self.input_files[0])
        print(sp)
        
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

            nevt_in, nevt_out = self._file_loop(writers)
        print(textwrap.dedent("""
                              Number of events in : {}
                              Number of events out: {}
                              Ratio               : {}
                              """.format(nevt_in, nevt_out, nevt_out / nevt_in)))
        return nevt_in, nevt_out

    def _file_loop(self, writers):
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
                    nevt_in, nevt_out, h5in, trigger_filter)

            if max_events_reached:
                print('Max events reached')
                break
            else:
                print("OK")

        return nevt_in, nevt_out

    def _event_loop(self, NEVT, pmtrwf, sipmrwf, pmtblr, events_info,
                    write,
                    nevt_in, nevt_out, h5in, trigger_pass):
        max_events_reached = False
        for evt in range(NEVT):
            nevt_in += 1
            if self.max_events_reached(nevt_in):
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
