"""
code: diomira.py
description: simulation of the response of the energy and tracking planes
for the NEXT detector.
author: J.J. Gomez-Cadenas
IC core team: Jacek Generowicz, JJGC, G. Martinez, J.A. Hernando, J.M Benlloch
package: invisible cities. See release notes and licence
last changed: 09-11-2017
"""
from __future__ import print_function

import sys
from glob import glob
from time import time

import numpy as np
import tables as tb

import invisible_cities.core.tbl_functions as tbl
from   invisible_cities.core.configure import configure, print_configuration
from   invisible_cities.cities.base_cities import SensorResponseCity, SensorParam
from   invisible_cities.core.nh5 import FEE
from   invisible_cities.core.random_sampling \
         import NoiseSampler as SiPMsNoiseSampler
import invisible_cities.core.wfm_functions as wfm
from   invisible_cities.core.system_of_units_c import SystemOfUnits
units = SystemOfUnits()




class Diomira(SensorResponseCity):
    """
    The city of DIOMIRA simulates the response of the energy and
    traking plane sensors.

    """
    def __init__(self,
                 run_number     = 0,
                 files_in       = None,
                 file_out       = None,
                 sipm_noise_cut = 3 * units.pes):
        """
        -1. Inits the machine
        -2. Loads the data base to access calibration data
        -3. Sets all switches to default value
        """

        SensorResponseCity.__init__(self,
                                    run_number     = run_number,
                                    files_in       = files_in,
                                    file_out       = file_out,
                                    sipm_noise_cut = sipm_noise_cut)

    def run(self, nmax):
        """
        Run the machine
        nmax is the max number of events to run
        """

        # TODO replace IOError with IC Exceptions

        n_events_tot = 0
        # run the machine only if in a legal state
        if not self.input_files:
            raise IOError('must set input files before running')
        if len(self.input_files)    == 0:
            raise IOError('input file list is empty')
        if not self.output_file:
            raise IOError('must set output file before running')
        if not self.sipm_noise_cut:
            raise IOError('must set sipm_noise_cut before running')

        self.display_IO_info(nmax)

        # loop over input files
        first = False
        with tb.open_file(self.output_file, "w",
                          filters = tbl.filters(self.compression)) as h5out:

            for ffile in self.input_files:

                print("Opening file {}".format(ffile))
                filename = ffile

                with tb.open_file(filename, "r") as h5in:
                    pmtrd  = h5in.root.pmtrd
                    sipmrd = h5in.root.sipmrd

                    NEVT, NPMT, PMTWL = pmtrd.shape
                    PMTWL_FEE = int(PMTWL / self.FE_t_sample)
                    NEVENTS_DST, NSIPM, SIPMWL = sipmrd.shape
                    sensor_param = SensorParam(NPMT   = NPMT,
                                               PMTWL  = PMTWL_FEE,
                                               NSIPM  = NSIPM,
                                               SIPMWL = SIPMWL)

                    print("Events in file = {}".format(NEVENTS_DST))

                    if not first:
                        # print configuration, create vectors
                        # init SiPM noiser, store FEE table
                        print_configuration({"# PMT"        : NPMT,
                                             "PMT WL"       : PMTWL,
                                             "PMT WL (FEE)" : PMTWL_FEE,
                                             "# SiPM"       : NSIPM,
                                             "SIPM WL"      : SIPMWL})
                        # RD group
                        RD = h5out.create_group(h5out.root, "RD")
                        # MC group
                        MC = h5out.create_group(h5out.root, "MC")
                        # create a table to store Energy plane FEE
                        self.fee_table = h5out.create_table(MC, "FEE", FEE,
                                          "EP-FEE parameters",
                                          tbl.filters("NOCOMPR"))
                        # create vectors
                        self.pmtrwf = h5out.create_earray(
                                    RD,
                                    "pmtrwf",
                                    atom = tb.Int16Atom(),
                                    shape = (0, NPMT, PMTWL_FEE),
                                    expectedrows = nmax,
                                    filters = tbl.filters(self.compression))

                        self.pmtblr = h5out.create_earray(
                                    RD,
                                    "pmtblr",
                                    atom = tb.Int16Atom(),
                                    shape = (0, NPMT, PMTWL_FEE),
                                    expectedrows = nmax,
                                    filters = tbl.filters(self.compression))

                        self.sipmrwf = h5out.create_earray(
                                    RD,
                                    "sipmrwf",
                                    atom = tb.Int16Atom(),
                                    shape = (0, NSIPM, SIPMWL),
                                    expectedrows = nmax,
                                    filters = tbl.filters(self.compression))

                        # Create instance of the noise sampler
                        self.noise_sampler = SiPMsNoiseSampler(SIPMWL, True)
                        # thresholds in adc counts
                        self.sipms_thresholds = (self.sipm_noise_cut
                                              *  self.sipm_adc_to_pes)
                        # store FEE parameters in table
                        self.store_FEE_table()

                        first = True

                    # loop over events in the file. Break when nmax is reached
                    for evt in range(NEVT):
                        # simulate PMT and SiPM response
                        # RWF and BLR
                        dataPMT, blrPMT = self.simulate_pmt_response(
                                        evt, pmtrd)
                        # append the data to pytable vectors
                        self.pmtrwf.append(
                            dataPMT.astype(int).reshape(1, NPMT, PMTWL_FEE))
                        self.pmtblr.append(
                             blrPMT.astype(int).reshape(1, NPMT, PMTWL_FEE))
                        # SiPMs
                        dataSiPM = self.simulate_sipm_response(
                                    evt, sipmrd, self.noise_sampler)
                        # return a noise-supressed waveform
                        dataSiPM = wfm.noise_suppression(
                                    dataSiPM, self.sipms_thresholds)

                        self.sipmrwf.append(
                            dataSiPM.astype(int).reshape(1, NSIPM, SIPMWL))

                        n_events_tot +=1
                        if n_events_tot % self.nprint == 0:
                            print('event in file = {}, total = {}'
                                  .format(evt, n_events_tot))

                        if n_events_tot >= nmax and nmax > -1:
                            print('reached maximum number of events (={})'
                                  .format(nmax))
                            break

        self.pmtrwf.flush()
        self.sipmrwf.flush()
        self.pmtblr.flush()


        return n_events_tot


def DIOMIRA(argv=sys.argv):
    """DIOMIRA DRIVER"""

    CFP = configure(argv)
    fpp = Diomira()
    files_in = glob(CFP['FILE_IN'])
    files_in.sort()
    fpp.set_input_files(files_in)
    fpp.set_output_file(CFP['FILE_OUT'])
    fpp.set_compression(compression=CFP['COMPRESSION'])
    fpp.set_print(nprint=CFP['NPRINT'])

    fpp.set_sipm_noise_cut(noise_cut = CFP["NOISE_CUT"])

    nevts = CFP['NEVENTS'] if not CFP['RUN_ALL'] else -1
    t0 = time()
    nevt = fpp.run(nmax=nevts)
    t1 = time()
    dt = t1 - t0

    print("run {} evts in {} s, time/event = {}".format(nevt, dt, dt/nevt))


if __name__ == "__main__":
    DIOMIRA(sys.argv)
