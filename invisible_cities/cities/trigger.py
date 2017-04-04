"""
code: trigger.py
description: simulation of trigger for the NEXT detector.
author: P. Ferrario
IC core team: Jacek Generowicz, JJGC, G. Martinez, J.A. Hernando, J.M Benlloch, P. Ferrario
package: invisible cities. See release notes and licence
last changed: 01-02-2017
"""

from __future__ import print_function

import sys
from glob import glob
from time import time

import numpy as np
import tables as tb

from invisible_cities.core.configure import configure, print_configuration
import invisible_cities.reco.tbl_functions as tbl
from invisible_cities.database import load_db
import invisible_cities.reco.peak_functions_c as cpf
from invisible_cities.cities.base_cities import City
import invisible_cities.database.load_db as db

class Trigger(City):
    """
    The city of TRIGGER simulates the trigger.

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
        self.trigger_channels = tr_channels
        self.min_height = minH*dataMC
        self.max_height = maxH*dataMC
        self.min_charge = minQ*dataMC
        self.max_charge = maxQ*dataMC
        self.min_width = minW
        self.max_width = maxW
        
    def run(self, nmax, run_number):
        """
        Run the machine
        nmax is the max number of events to run
        """
        n_events_tot = 0
        # run the machine only if in a legal state
        self.check_files()

        with tb.open_file(self.output_file, "w",
                          filters = tbl.filters(self.compression)) as h5out:
            first=False
            for cfile in self.input_files:

                print("Opening file {}".format(cfile))
                try:
                    with tb.open_file(cfile, "r+") as h5in:
                        pmtcwf  = h5in.root.BLR.pmtcwf
                        sipmrwf = h5in.root.BLR.sipmrwf
                        nevts_pmt, npmts, pmt_wndw_length = pmtcwf.shape
                        nevts_sipm, nsipms, sipm_wndw_length = sipmrwf.shape

                        if first == False:                       
                            # create vectors                        
                            self.pmttrigwf = h5out.create_earray(
                                        h5out.root,
                                        "pmttrigwf",
                                        atom=tb.Int16Atom(),
                                        shape=(0, npmts, pmt_wndw_length),
                                        expectedrows=nmax,
                                        filters=tbl.filters(self.compression))
                            self.sipmtrigwf = h5out.create_earray(
                                        h5out.root,
                                        "sipmtrigwf",
                                        atom=tb.Int16Atom(),
                                        shape=(0, nsipms, sipm_wndw_length),
                                        expectedrows=nmax,
                                        filters=tbl.filters(self.compression))
                            first = True
                        
                        dataPMT = db.DataPMT(run_number)
                    
                        evt = 0
     #                   while evt < nmax: 
                        for evt in range(nevts_pmt):
                            self.event_in += 1
                            ok_channels = 0
                            for pmt in range(npmts):
     #                           print('PMT %s'%(pmt))
                                found = False
                                if (dataPMT.SensorID[pmt] in self.trigger_channels):
                                    pk_content, pk_indx = cpf.wfzs(pmtcwf[evt, pmt].astype(np.double), threshold=self.min_height)
                                    pks = cpf.find_S12(pk_content, pk_indx)
      #                              print(len(pks))
                                    for value in pks.values():
                                        width = value[0][-1]-value[0][0]
                                        charge = np.sum(value[1])
                                        height = np.amax(value[1])
                                        if width > self.min_width and width < self.max_width:
                                            if charge > self.min_charge and charge < self.max_charge:
                                                if height > self.min_height and height < self.max_height:
                                                    found = True
                                    if found:
                                        ok_channels += 1
                            if ok_channels >= self.min_trigger_channels:
                                self.pmttrigwf.append(np.array(pmtcwf[evt]).astype(np.double).reshape(1, npmts, pmt_wndw_length))
                                self.sipmtrigwf.append(np.array(sipmrwf[evt]).astype(np.double).reshape(1, nsipms, sipm_wndw_length))
                                self.event_out += 1
                            evt+=1
                            n_events_tot +=1
                            if evt%self.nprint == 0:
                                print('event number = {}, total = {}'.\
                                  format(evt, n_events_tot))
                            if n_events_tot >= nmax and nmax > -1:
                                print('reached maximum number of events (={})'.\
                                      format(nmax))
                                break
                                    
                    
                except:
                    print('Error: input file cannot be opened')
                    raise

       
                        
        return n_events_tot


def TRIGGER(argv=sys.argv):
    """TRIGGER DRIVER"""

    CFP = configure(argv)
    fpp = Trigger()
    files_in = glob(CFP.FILE_IN)
    files_in.sort()
    fpp.set_input_files(files_in)
    fpp.set_output_file(CFP.FILE_OUT)
    fpp.set_compression(CFP.COMPRESSION)
    fpp.set_print(nprint=CFP.NPRINT)

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
    
    fpp.set_trigger_conf(number, tr_ch, minH, maxH, minQ, maxQ, minW, maxW, dataMC)

    nevts = CFP.NEVENTS if not CFP.RUN_ALL else -1
    t0 = time()
    nevt = fpp.run(nmax=nevts, run_number=0)
    t1 = time()
    dt = t1 - t0

    print("run {} evts in {} s, time/event = {}".format(nevt, dt, dt/nevt))


if __name__ == "__main__":
    TRIGGER(sys.argv)
