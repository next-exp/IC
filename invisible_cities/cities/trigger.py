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

class Trigger:
    """
    The city of TRIGGER simulates the trigger.

    """
    def __init__(self):
        """
        - Sets default values for state attributes
        """
    # Set the machine default state
        self.set_input_files_    = False
        self.set_output_file_    = False


    def set_input_files(self, input_files):
        """Set the input files."""
        self.input_files = input_files
        self.set_input_files_ = True

    def set_output_file(self, output_file, compression='ZLIB4'):
        """Set the output file."""
 #       filename = output_file
        self.compression = compression
        self.h5out = tb.open_file(output_file, "w",
                          filters=tbl.filters(compression))
        self.set_output_file_ = True
        
    def run(self, nmax, tr_type):
        """
        Run the machine
        nmax is the max number of events to run
        """
        n_events_tot = 0
        # run the machine only if in a legal state
        if self.set_input_files_    == False:
            raise IOError('must set input files before running')
        if len(self.input_files)    == 0:
            raise IOError('input file list is empty')
        if self.set_output_file_    == False:
            raise IOError('must set output file before running')

        first=False
        for cfile in self.input_files:

            print("Opening file {}".format(cfile))
 #           filename = cfile
            try:
                with tb.open_file(cfile, "r+") as h5in:
                    pmtrd  = h5in.root.pmtrd
                    nevts_pmt, npmts, pmt_wndw_length = pmtrd.shape
                    nevts_sipm, nsipms, sipm_wndw_length = sipmrd.shape

                    if first == False:                       
                        # create vectors
                        self.pmtrwf = self.h5out.create_earray(
                                    self.h5out.root,
                                    "pmttrigwf",
                                    atom=tb.Int16Atom(),
                                    shape=(0, npmts, pmt_wndw_length),
                                    expectedrows=nmax,
                                    filters=tbl.filters(self.compression))
                        self.sipmrwf = self.h5out.create_earray(
                                    self.h5out.root,
                                    "sipmtrigwf",
                                    atom=tb.Int16Atom(),
                                    shape=(0, nsipms, sipm_wndw_length),
                                    expectedrows=nmax,
                                    filters=tbl.filters(self.compression))
                        first = True
                        
                    evt = 0
                    while evt < nevts_pmt:

        return n_events_tot


def TRIGGER(argv=sys.argv):
    """TRIGGER DRIVER"""

    CFP = configure(argv)
    fpp = Trigger()
    files_in = glob(CFP['FILE_IN'])
    files_in.sort()
    fpp.set_input_files(files_in)
    fpp.set_output_file(CFP['FILE_OUT'],
                        compression=CFP['COMPRESSION'])
    fpp.set_print(nprint=CFP['NPRINT'])

    trigger_type = CFP['TRIGGER']

    nevts = CFP['NEVENTS'] if not CFP['RUN_ALL'] else -1
    t0 = time()
    nevt = fpp.run(nmax=nevts, tr_type=trigger_type)
    t1 = time()
    dt = t1 - t0

    print("run {} evts in {} s, time/event = {}".format(nevt, dt, dt/nevt))


if __name__ == "__main__":
    TRIGGER(sys.argv)
