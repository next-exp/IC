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
from invisible_cities.core.configure import configure, print_configuration
from invisible_cities.core.nh5 import FEE
import invisible_cities.core.mpl_functions as mpl
import invisible_cities.core.wfm_functions as wfm
from invisible_cities.core.random_sampling\
     import NoiseSampler as SiPMsNoiseSampler
from invisible_cities.database import load_db
import invisible_cities.sierpe.fee as FE


class Diomira:
    """
    The city of DIOMIRA simulates the response of the energy and
    traking plane sensors.

    """
    def __init__(self):
        """
        -1. Inits the machine
        -2. Loads the data base to access calibration data
        -3. Sets all switches to default value
        """
        #access data base
        DataPMT = load_db.DataPMT(0)  # argument = 0 implies MC
        DataSiPM = load_db.DataSiPM(0)
        # number of adc counts per PES
        self.adc_to_pes = abs(DataPMT.adc_to_pes.values).astype(np.double)
        self.sipm_adc_to_pes = DataSiPM.adc_to_pes.values.astype(np.double)
        # deconvolution coefficientes
        self.coeff_c = DataPMT.coeff_c.values.astype(np.double)
        self.coeff_blr = DataPMT.coeff_blr.values.astype(np.double)
        # PMT noise rms
        self.noise_rms = DataPMT.noise_rms.values.astype(np.double)
        # Set the machine default state
        self.set_input_files_    = False
        self.set_output_file_    = False
        self.set_sipm_noise_cut_ = False
        self.plot_raw = False
        self.plot_blr = False
        self.plot_rwf = False

    def simulate_sipm_response(self, event, sipmrd,
                               sipms_noise_sampler):
        """Add noise with the NoiseSampler class and return
        the noisy waveform (in pes)."""
        # add noise (in PES) to true waveform
        dataSiPM = sipmrd[event] + sipms_noise_sampler.Sample()
        # return total signal in adc counts
        return wfm.to_adc(dataSiPM, self.sipm_adc_to_pes)

    def simulate_pmt_response(self, event, pmtrd):
        """ Full simulation of the energy plane response
        Input:
         1) extensible array pmtrd
         2) event_number

        returns:
        array of raw waveforms (RWF) obtained by convoluting pmtrd with the PMT
        front end electronics (LPF, HPF filters)
        array of BLR waveforms (only decimation)
        """
        # Single Photoelectron class
        spe = FE.SPE()
        # FEE, with noise PMT
        fee = FE.FEE(noise_FEEPMB_rms=FE.NOISE_I, noise_DAQ_rms=FE.NOISE_DAQ)
        NPMT = pmtrd.shape[1]
        RWF = []
        BLRX = []

        for pmt in range(NPMT):
            # normalize calibration constants from DB to MC value
            cc = self.adc_to_pes[pmt] / FE.ADC_TO_PES
            # signal_i in current units
            signal_i = FE.spe_pulse_from_vector(spe, pmtrd[event, pmt])
            # Decimate (DAQ decimation)
            signal_d = FE.daq_decimator(FE.f_mc, FE.f_sample, signal_i)
            # Effect of FEE and transform to adc counts
            signal_fee = FE.signal_v_fee(fee, signal_d, pmt) * FE.v_to_adc()
            # add noise daq
            signal_daq = cc * FE.noise_adc(fee, signal_fee)
            # signal blr is just pure MC decimated by adc in adc counts
            signal_blr = cc * FE.signal_v_lpf(fee, signal_d)*FE.v_to_adc()
            # raw waveform stored with negative sign and offset
            RWF.append(FE.OFFSET - signal_daq)
            # blr waveform stored with positive sign and no offset
            BLRX.append(signal_blr)
        return np.array(RWF), np.array(BLRX)

    def set_plot(self, plot_raw=False, plot_blr=False,
                 plot_rwf=False, w_size=800):
        """Decide what to plot."""
        self.plot_raw     = plot_raw
        self.plot_blr     = plot_blr
        self.plot_rwf     = plot_rwf
        self.w_size       = w_size

    def set_print(self, nprint=10):
        """Print frequency."""
        self.nprint = nprint

    def set_input_files(self, input_files):
        """Set the input files."""
        self.input_files = input_files
        self.set_input_files_ = True

    def set_output_file(self, output_file, compression='ZLIB4'):
        """Set the output file."""
        filename = output_file
        self.compression = compression
        self.h5out = tb.open_file(filename, "w",
                          filters=tbl.filters(compression))
        self.set_output_file_ = True

    def set_sipm_noise_cut(self, noise_cut=3.0):
        """Sets the SiPM noise cut (in PES)"""
        self.sipm_noise_cut = noise_cut
        self.set_sipm_noise_cut_ = True

    def store_FEE_table(self):
        """Store the parameters of the EP FEE simulation."""
        row = self.fee_table.row
        row["OFFSET"] = FE.OFFSET
        row["CEILING"] = FE.CEILING
        row["PMT_GAIN"] = FE.PMT_GAIN
        row["FEE_GAIN"] = FE.FEE_GAIN
        row["R1"] = FE.R1
        row["C1"] = FE.C1
        row["C2"] = FE.C2
        row["ZIN"] = FE.Zin
        row["DAQ_GAIN"] = FE.DAQ_GAIN
        row["NBITS"] = FE.NBITS
        row["LSB"] = FE.LSB
        row["NOISE_I"] = FE.NOISE_I
        row["NOISE_DAQ"] = FE.NOISE_DAQ
        row["t_sample"] = FE.t_sample
        row["f_sample"] = FE.f_sample
        row["f_mc"] = FE.f_mc
        row["f_LPF1"] = FE.f_LPF1
        row["f_LPF2"] = FE.f_LPF2
        row["coeff_c"] = self.coeff_c
        row["coeff_blr"] = self.coeff_blr
        row["adc_to_pes"] = self.adc_to_pes
        row["pmt_noise_rms"] = self.noise_rms
        row.append()
        self.fee_table.flush()

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
        if self.set_sipm_noise_cut_ == False:
            raise IOError('must set sipm_noise_cut before running')

        print("""
                 DIOMIRA will run a max of {} events
                 Input Files = {}
                 Output File = {}
                          """.format(nmax,
                                     self.input_files,
                                     self.h5out))

        # loop over input files
        first=False
        for ffile in self.input_files:

            print("Opening file {}".format(ffile))
            filename = ffile
            try:
                with tb.open_file(filename, "r+") as h5in:
                    pmtrd  = h5in.root.pmtrd
                    sipmrd = h5in.root.sipmrd

                    NEVT, NPMT, PMTWL = pmtrd.shape
                    PMTWL_FEE = int(PMTWL/FE.t_sample)
                    NEVENTS_DST, NSIPM, SIPMWL = sipmrd.shape

                    print("Events in file = {}".format(NEVENTS_DST))

                    if first == False:
                        # print configuration, create vectors
                        # init SiPM noiser, store FEE table
                        print_configuration({"# PMT": NPMT, "PMT WL": PMTWL,
                                             "PMT WL (FEE)": PMTWL_FEE,
                                             "# SiPM": NSIPM,
                                             "SIPM WL": SIPMWL})
                        # RD group
                        RD = self.h5out.create_group(self.h5out.root, "RD")
                        # MC group
                        MC = self.h5out.create_group(self.h5out.root, "MC")
                        # create a table to store Energy plane FEE
                        self.fee_table = self.h5out.\
                             create_table(MC, "FEE", FEE,
                                          "EP-FEE parameters",
                                          tbl.filters("NOCOMPR"))
                        # create vectors
                        self.pmtrwf = self.h5out.create_earray(
                                    RD,
                                    "pmtrwf",
                                    atom=tb.Int16Atom(),
                                    shape=(0, NPMT, PMTWL_FEE),
                                    expectedrows=nmax,
                                    filters=tbl.filters(self.compression))

                        self.pmtblr = self.h5out.create_earray(
                                    RD,
                                    "pmtblr",
                                    atom=tb.Int16Atom(),
                                    shape=(0, NPMT, PMTWL_FEE),
                                    expectedrows=nmax,
                                    filters=tbl.filters(self.compression))

                        self.sipmrwf = self.h5out.create_earray(
                                    RD,
                                    "sipmrwf",
                                    atom=tb.Int16Atom(),
                                    shape=(0, NSIPM, SIPMWL),
                                    expectedrows=nmax,
                                    filters=tbl.filters(self.compression))

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

                        # optional plotting of data
                        if self.plot_raw:
                            print('plot raw')
                            mpl.plot_waveforms(pmtrd[evt], maxlen=0, zoom=True,
                                                window_size=self.w_size*40)
                            plt.show()
                            raw_input('->')

                        if self.plot_rwf:
                            print('plot rwf')
                            mpl.plot_waveforms(dataPMT, maxlen=0, zoom=True,
                                                window_size=self.w_size)
                            plt.show()
                            raw_input('->')

                        if self.plot_blr:
                            print('plot blr')
                            mpl.plot_waveforms(blrPMT, maxlen=0, zoom=True,
                                                window_size=self.w_size)
                            plt.show()
                            raw_input('->')

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


        self.pmtrwf.flush()
        self.sipmrwf.flush()
        self.pmtblr.flush()
        self.h5out.close()

        return n_events_tot


def DIOMIRA(argv=sys.argv):
    """DIOMIRA DRIVER"""

    CFP = configure(argv)
    fpp = Diomira()
    files_in = glob(CFP['FILE_IN'])
    files_in.sort()
    fpp.set_input_files(files_in)
    fpp.set_output_file(CFP['FILE_OUT'],
                        compression=CFP['COMPRESSION'])
    fpp.set_print(nprint=CFP['NPRINT'])

    fpp.set_sipm_noise_cut(noise_cut=CFP["NOISE_CUT"])

    nevts = CFP['NEVENTS'] if not CFP['RUN_ALL'] else -1
    t0 = time()
    nevt = fpp.run(nmax=nevts)
    t1 = time()
    dt = t1 - t0

    print("run {} evts in {} s, time/event = {}".format(nevt, dt, dt/nevt))


if __name__ == "__main__":
    DIOMIRA(sys.argv)
