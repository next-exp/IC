"""
code: cecilia.py
description: simulation of trigger for the NEXT detector.
author: P. Ferrario
IC core team: Jacek Generowicz, JJGC, G. Martinez, J.A. Hernando, J.M Benlloch
package: invisible cities. See release notes and licence
last changed: 01-02-2017
"""

import sys
from glob import glob
from time import time
from textwrap import dedent

import numpy as np
import tables as tb

from .. core.configure import configure
from .. core.configure import print_configuration
from .. reco           import tbl_functions    as tbl
from .. reco           import peak_functions_c as cpf
from .. database       import load_db          as db
from .. reco.nh5       import RunInfo
from .. reco.nh5       import EventInfo
from .  base_cities    import City
from .  base_cities    import merge_two_dicts


class Cecilia(City):
    """
    The city of CECILIA simulates the trigger.

    """
    def __init__(self):
        """
        - Sets default values for state attributes
        """
    # Set the machine default state
        self.event_in = 0
        self.event_out = 0

    def set_trigger_conf(self, number, tr_channels, minH, maxH, minQ, maxQ, minW, maxW, dataMC):
        """Set the parameters of the trigger configuration."""
        self.min_trigger_channels = number
        self.    trigger_channels = tr_channels
        self.min_height = minH * dataMC
        self.max_height = maxH * dataMC
        self.min_charge = minQ * dataMC
        self.max_charge = maxQ * dataMC
        self.min_width = minW
        self.max_width = maxW

    def run(self, nmax, first_evt, run_number):
        """
        Run the machine
        nmax is the max number of events to run
        """
        def write_evt():
            self. pmttrigwf.append(np.array( pmtcwf[evt]).astype(np.double).reshape(1,  npmts,  pmt_wndw_length))
            self.sipmtrigwf.append(np.array(sipmrwf[evt]).astype(np.double).reshape(1, nsipms, sipm_wndw_length))
            evt_row = self.eventInfo.row
            evt_row['evt_number'] = evt + first_evt
            evt_row['timestamp'] = 0
            evt_row.append()
            self.event_out += 1

        n_events_tot = 0
        # run the machine only if in a legal state
        self.check_files()

        with tb.open_file(self.output_file, "w",
                          filters = tbl.filters(self.compression)) as h5out:
            first=False
            for cfile in self.input_files:

                print("Opening file {}".format(cfile))

                with tb.open_file(cfile, "r+") as h5in:
                    pmtcwf  = h5in.root.BLR.pmtcwf
                    sipmrwf = h5in.root.BLR.sipmrwf
                    nevts_pmt, npmts, pmt_wndw_length = pmtcwf.shape
                    nevts_sipm, nsipms, sipm_wndw_length = sipmrwf.shape
 #                   print(nevts_pmt, nevts_sipm)

                    if first == False:
                        # create vectors
                        self.pmttrigwf = h5out.create_earray(
                                    h5out.root,
                                    "pmttrigwf",
                                    atom         = tb.Int16Atom(),
                                    shape        = (0, npmts, pmt_wndw_length),
                                    expectedrows = nmax,
                                    filters      = tbl.filters(self.compression))
                        self.sipmtrigwf = h5out.create_earray(
                                    h5out.root,
                                    "sipmtrigwf",
                                    atom         = tb.Int16Atom(),
                                    shape        = (0, nsipms, sipm_wndw_length),
                                    expectedrows = nmax,
                                    filters      = tbl.filters(self.compression))
                        run_g = h5out.create_group(h5out.root, 'Run')
                        self.eventInfo = h5out.create_table(
                                    run_g,
                                    "events",
                                    EventInfo,
                                    "Events info",
                                    tbl.filters("NOCOMPR"))
                        self.runInfo = h5out.create_table(
                                    run_g,
                                    "run",
                                    RunInfo,
                                    "Run info",
                                    tbl.filters("NOCOMPR"))
                        first = True

                    dataPMT = db.DataPMT(run_number)


                    evt = 0
                    for evt in range(nevts_pmt):
                        self.event_in += 1
                        ok_channels = 0
                        for pmt in range(npmts):
     #                       print('PMT %s'%(pmt))
                            if (dataPMT.SensorID[pmt] in self.trigger_channels):
                                if self._pass_trigger(evt, pmtcwf, pmt):
                                    ok_channels += 1
                        if ok_channels >= self.min_trigger_channels:
                            write_evt()
                        n_events_tot += 1
                        self.show_progress(evt, n_events_tot)
                        if self.terminate(n_events_tot, nmax):
                            break
  #                      print(len(self.pmttrigwf), len(self.sipmtrigwf))

        print('Trigger city: events in = {}'.format(self.event_in))
        print('Trigger city: events out = {}'.format(self.event_out))
        return n_events_tot

    config_file_format = City.config_file_format + """

# run
NEVENTS {NEVENTS}
RUN_ALL {RUN_ALL}
FIRST_EVT {FIRST_EVT}
RUN_NUMBER {RUN_NUMBER}

# set_print
NPRINT {NPRINT}

##specific
TR_CHANNELS {TR_CHANNELS}
MIN_NUMB_CHANNELS {MIN_NUMB_CHANNELS}
MIN_HEIGHT {MIN_HEIGHT}
MAX_HEIGHT {MAX_HEIGHT}
MIN_CHARGE {MIN_CHARGE}
MAX_CHARGE {MAX_CHARGE}
MIN_WIDTH {MIN_WIDTH}
MAX_WIDTH {MAX_WIDTH}
DATA_MC_RATIO {DATA_MC_RATIO}"""

    config_file_format = dedent(config_file_format)

    default_config = merge_two_dicts(
        City.default_config,
        dict(FILE_IN  = None,
             COMPRESSION = 'ZLIB4',
             NEVENTS = -1,
             RUN_ALL = False,
             FIRST_EVT = 0,
             RUN_NUMBER = 0,
             NPRINT = 10,
             TR_CHANNELS = '0 1 2 3 4 5 6 7 8 9 10 11',
             MIN_NUMB_CHANNELS = 5,
             MIN_HEIGHT = 15,
             MAX_HEIGHT = 1000,
             MIN_CHARGE = 3000,
             MAX_CHARGE = 20000,
             MIN_WIDTH = 4000,
             MAX_WIDTH = 12000,
             DATA_MC_RATIO = 0.8))

    def _pass_trigger(self, evt, pmtcwf, pmt):
        found = False
        pk_content, pk_indx = cpf.wfzs(pmtcwf[evt, pmt].astype(np.double), threshold=self.min_height)
        pks = cpf.find_S12(pk_content, pk_indx)
    #                              print(len(pks))
        for value in pks.values():
            width = value[0][-1] - value[0][0]
            charge = np.sum(value[1])
            height = np.max(value[1])
            if self.min_width < width < self.max_width:
                if self.min_charge < charge < self.max_charge:
                    if self.min_height < height < self.max_height:
                        found = True
        return found

    def show_progress(self, evt, n_events_tot):
        if evt % self.nprint == 0:
            print('event number = {evt}, total = {n_events_tot}'.format(**locals()))

    def terminate(self, n_events_tot, nmax):
        if n_events_tot >= nmax and nmax > -1:
            print('maximum number of events reached (={})'.format(nmax))
            return True
        return False


def CECILIA(argv=sys.argv):
    """CECILIA DRIVER"""

    CFP = configure(argv)
    tr = Cecilia()
    files_in = glob(CFP.FILE_IN)
    files_in.sort()
    tr.set_input_files(files_in)
    tr.set_output_file(CFP.FILE_OUT)
    tr.set_compression(CFP.COMPRESSION)
    tr.set_print(nprint=CFP.NPRINT)

    ## Trigger configuration parameters
    tr_ch = CFP.TR_CHANNELS
    number = CFP.MIN_NUMB_CHANNELS
    minH = CFP.MIN_HEIGHT
    maxH = CFP.MAX_HEIGHT
    minQ = CFP.MIN_CHARGE
    maxQ = CFP.MAX_CHARGE
    minW = CFP.MIN_WIDTH
    maxW = CFP.MAX_WIDTH
    dataMC = CFP.DATA_MC_RATIO

 #   trigger_type = CFP.TRIGGER

    tr.set_trigger_conf(number, tr_ch, minH, maxH, minQ, maxQ, minW, maxW, dataMC)

    nevts = CFP.NEVENTS if not CFP.RUN_ALL else -1
    t0 = time()
    n_evts_run = tr.run(nmax=nevts, first_evt=CFP.FIRST_EVT, run_number=CFP.RUN_NUMBER)
    t1 = time()
    dt = t1 - t0

    print("run {} evts in {} s, time/event = {}".format(n_evts_run, dt, dt/n_evts_run))

    return nevts, n_evts_run


if __name__ == "__main__":
    CECILIA(sys.argv)
