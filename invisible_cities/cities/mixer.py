"""
code: mixer.py
description: simulation of pile-up for the NEXT detector.
author: P. Ferrario
IC core team: Jacek Generowicz, JJGC, G. Martinez, J.A. Hernando, J.M Benlloch, P. Ferrario
package: invisible cities. See release notes and licence
last changed: 01-02-2017
"""

from __future__ import print_function

import sys
from glob import glob
from time import time
from timeit import default_timer as timer

import numpy as np
import tables as tb
import random as rand
from scipy.ndimage.interpolation import shift

from invisible_cities.core.configure import configure, print_configuration
import invisible_cities.core.tbl_functions as tbl

class Mixer:
    """
    The city of MIXER simulates pile-up.

    """
    def __init__(self, mean_evts_per_wndw, s1_pos):
        """
        - Sets default values for state attributes
        """
        self.mean_evts_per_wndw = mean_evts_per_wndw
        self.s1pos = s1_pos
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

    def set_print(self, nprint=10):
        """Print frequency."""
        self.nprint = nprint
        
    def run(self, nmax):
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
            try:
                with tb.open_file(cfile, "r+") as h5in:
                    pmtrd  = h5in.root.pmtrd
                    sipmrd = h5in.root.sipmrd
                    nevts_pmt, npmts, pmt_wndw_length = pmtrd.shape
                    nevts_sipm, nsipms, sipm_wndw_length = sipmrd.shape

                    if first == False:
                        
                        # RD group
                        RD = self.h5out.create_group(self.h5out.root, "RD")
                                        
                        # create vectors
                        self.pmtmwf = self.h5out.create_earray(
                                    RD,
                                    "pmtmwf",
                                    atom=tb.Int16Atom(),
                                    shape=(0, npmts, pmt_wndw_length),
                                    expectedrows=nmax,
                                    filters=tbl.filters(self.compression))
                        self.sipmmwf = self.h5out.create_earray(
                                    RD,
                                    "sipmwf",
                                    atom=tb.Int16Atom(),
                                    shape=(0, nsipms, sipm_wndw_length),
                                    expectedrows=nmax,
                                    filters=tbl.filters(self.compression))
                        first = True

                    evt = 0
                    while evt < nevts_pmt:  
                        ndecays = np.random.poisson(self.mean_evts_per_wndw)
                        print ('Number of decays = {}'.format(ndecays))
                        if ndecays > 1:
                            times = []
                            for d in range(ndecays):
                                times.append(rand.randrange(0, pmt_wndw_length))
                                print ('Time = {}'.format(rand.randrange(0, pmt_wndw_length)))
                            mwvfs = []
                            for pmt in range(npmts):
                                merged_wvf = shift(pmtrd[evt, pmt], times[0]-self.s1pos, cval=0)
                                for d in range(1,ndecays):
                                    merged_wvf = np.add(shift(pmtrd[evt+d, pmt], times[d]-self.s1pos, cval=0), merged_wvf)                                
                                mwvfs.append(merged_wvf)
                            self.pmtmwf.append(np.array(mwvfs).astype(int).reshape(1, npmts, pmt_wndw_length))
                        elif ndecays == 1:
                            self.pmtmwf.append(np.array(pmtrd[evt]).astype(int).reshape(1, npmts, pmt_wndw_length))
    
                        evt += ndecays
                        n_events_tot +=1
                        if n_events_tot%self.nprint == 0:
                            print('event in file = {}, total = {}'.\
                              format(evt, n_events_tot))

                        if n_events_tot >= nmax and nmax > -1:
                            print('reached maximum number of events (={})'.\
                                  format(nmax))
                            break
                        
            except:
                print('error')
                raise

        return n_events_tot


def MIXER(argv=sys.argv):
    """MIXER DRIVER"""

    CFP = configure(argv)
    fpp = Mixer(CFP['MEAN_EVTS_PER_WINDOW'], CFP['S1_POS'])
    files_in = glob(CFP['FILE_IN'])
    files_in.sort()
    fpp.set_input_files(files_in)
    fpp.set_output_file(CFP['FILE_OUT'],
                        compression=CFP['COMPRESSION'])

    fpp.set_print(nprint=CFP['NPRINT'])

    nevts = CFP['NEVENTS'] if not CFP['RUN_ALL'] else -1
    start = timer()
    nevt = fpp.run(nmax=nevts)
    
    dt = timer() - start

    print("run {} evts in {} s, time/event = {}".format(nevt, dt, dt/nevt))


if __name__ == "__main__":
    MIXER(sys.argv)
