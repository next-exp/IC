"""
DIOMIRA
JJGC August-October 2016
GML October 2016
New version, JJ, afer VHB, November 2016

What DIOMIRA does:
1) Reads a MCRD file containing MC waveforms for the 12 PMTs of the EP.
   Each waveform contains number of PEs in bins of 1 ns.
2) Convolves the PE waveform with the response of the FEE electronics.
3) Decimates the waveform, simulating the effect of the DAQ sampling
(25 ns bins)
4) Writes a RWF file with the new data and adds the FEE simulation parameters
as metadata
"""

from __future__ import print_function
import sys
import numpy as np
import tables
from time import time

import Core.system_of_units as units
from Core.LogConfig import logger
from Core.Configure import configure, define_event_loop, print_configuration
from Core.Nh5 import FEE, SENSOR_WF
import Core.wfmFunctions as wfm
import Core.coreFunctions as cf
import Core.tblFunctions as tbl
from Core.RandomSampling import NoiseSampler as SiPMsNoiseSampler

import Sierpe.FEE as FE
import Database.load_db as DB


"""

DIOMIRA
ChangeLog:

26.9

Changed types of PMTRWF, SIPMRWF and PMTTWF to Float32 for
    (compatibility with ART/GATE)

Do not store EPMT and ESIPM (can be computed on the fly)

Change sign of pmtrwf to negative (as produced by the DAQ)

28.9 add cython

29.9 changed the way true waveforms are treated.
before: --> full waveform (in bins of 25 ns) in an EArray
now--> ZS waveform rebinned at 1 mus in a Table
(adavantages:) faster processing less space

01.10 moved loop definition to Configure.py and added index to TWF table

11.10 introduced SiPM noise. SiPMs' true waveforms stored under TWF group.
Some variables, classes and functions renamed for clarity.

12.10 ZS functions to store the SiPMs

13.10 Reutilization of functions and some duplicities removed.
      SiPMs' waveforms to be stored without ZS.

14.10 JJ: Store all waveforms as Int16 (and as adc counts)

17.10: JJ PEP8 compliant and pyflakes compliant
(adjust to PEP8 syntax, eliminate import* etc.)

18.10 GML, add soft cut to eliminate noise below 0.5 pes in sipm plane

19.10 JJ: write calibration constants to FEE table!

20.10: JJ, store BLR with positive signal and baseline subtracted

20.10: GML, overwrite calibration constants in DataPMT with values
from FEE table. PRE-RELEASE

15.11, new version of FEE for PMTs

16.11: Using new database utility
"""


def simulate_sipm_response(event_number, sipmrd_, sipms_noise_sampler):
    """Add noise with the NoiseSampler class and return the noisy waveform."""
    return sipmrd_[event_number] + sipms_noise_sampler.Sample()


def simulate_pmt_response(event, pmtrd):
    """
    Input:
     1) extensible array pmtrd
     2) event_number

    returns:
    array of raw waveforms (RWF), obtained by convoluting pmtrd_ with the PMT
    front end electronics (LPF, HPF)
    array of BLR waveforms (only decimation)
    """

    spe = FE.SPE()  # spe
    # FEE, with noise PMT
    fee = FE.FEE(noise_FEEPMB_rms=FE.NOISE_I, noise_DAQ_rms=FE.NOISE_DAQ)
    NPMT = pmtrd.shape[1]
    RWF = []
    BLRX = []
    DataPMT = DB.DataPMT(0)  # run_number = 0 describes MC data
    adc_to_pes = np.abs(DataPMT.adc_to_pes.values)
    for pmt in range(NPMT):
        # signal_i in current units
        cc = adc_to_pes[pmt] / FE.ADC_TO_PES
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
        BLRX.append(FE.OFFSET - signal_blr)
    return np.array(RWF), np.array(BLRX)


def DIOMIRA(argv=sys.argv):
    """Diomira driver."""
    CFP = configure(argv)

    if CFP["INFO"]:
        print("""
        DIOMIRA:
         1. Reads a MCRD file produced by art/centella, which stores MCRD
        waveforms for PMTs (bins of 1 ns) and SiPMs (bins of 1 mus)
        2. Simulates the response of the energy plane and outputs both RWF
        and TWF
        3. Simulates the response of the tracking plane in the SiPMs and
        outputs SiPM RWF
        4. Add a table describing the FEE parameters used for simulation
        5. Copies the tables on geometry, detector data and MC
        """)

    # open the input file
    with tables.open_file(CFP["FILE_IN"], "r") as h5in:
        # access the PMT raw data in file
        pmtrd_ = h5in.root.pmtrd
        sipmrd_ = h5in.root.sipmrd

        NEVENTS_DST, NPMT, PMTWL = pmtrd_.shape
        PMTWL_FEE = int(PMTWL/FE.t_sample)
        NEVENTS_DST, NSIPM, SIPMWL = sipmrd_.shape

        print_configuration({"# PMT": NPMT, "PMT WL": PMTWL,
                             "PMT WL (FEE)": PMTWL_FEE,
                             "# SiPM": NSIPM, "SIPM WL": SIPMWL,
                             "# events in DST": NEVENTS_DST})

        # access the geometry and the sensors metadata info
        mctrk_t = h5in.root.MC.MCTracks
        sipmdf = DB.DataSiPM(0)  # run_number = 0 for MC info

        # Create instance of the noise sampler
        noise_sampler_ = SiPMsNoiseSampler(SIPMWL, True)
        sipms_thresholds_ = CFP["NOISE_CUT"] * np.array(sipmdf["adc_to_pes"])

        COMPRESSION = CFP["COMPRESSION"]
        # open the output file
        with tables.open_file(CFP["FILE_OUT"], "w",
                              filters=tbl.filters(COMPRESSION)) as h5out:

            # create a group to store sensors IDs
            sgroup = h5out.create_group(h5out.root, "Sensors")
            # copy DataSiPM & DataPMT table
            h5in.root.Sensors.DataSiPM.copy(newparent=sgroup)
            h5in.root.Sensors.DataPMT.copy(newparent=sgroup)

            # create a group to store MC data
            mcgroup = h5out.create_group(h5out.root, "MC")
            # copy the mctrk table
            mctrk_t.copy(newparent=mcgroup)

            # create a table to store Energy plane FEE, hang it from MC group
            fee_table = h5out.create_table(mcgroup, "FEE", FEE,
                                           "EP-FEE parameters",
                                           tbl.filters("NOCOMPR"))

            # # create a group to store True waveform data
            # twfgroup = h5out.create_group(h5out.root, "TWF")
            # # create a table to store true waveform (zs, rebinned)
            # pmt_twf_table = h5out.create_table(twfgroup, "PMT", SENSOR_WF,
            #                                    "Store for PMTs TWF",
            #                                    tbl.filters(COMPRESSION))
            #
            # sipm_twf_table = h5out.create_table(twfgroup, "SiPM", SENSOR_WF,
            #                                     "Store for SiPM TWF",
            #                                     tbl.filters(COMPRESSION))
            #
            # # and index in event column
            # pmt_twf_table.cols.event.create_index()
            # sipm_twf_table.cols.event.create_index()

            # fill FEE table
            tbl.store_FEE_table(fee_table)

            # create a group to store RawData
            h5out.create_group(h5out.root, "RD")

            # create an extensible array to store the RWF waveforms
            pmtrwf = h5out.create_earray(h5out.root.RD, "pmtrwf",
                                         atom=tables.Int16Atom(),
                                         shape=(0, NPMT, PMTWL_FEE),
                                         expectedrows=NEVENTS_DST,
                                         filters=tbl.filters(COMPRESSION))

            pmtblr = h5out.create_earray(h5out.root.RD, "pmtblr",
                                         atom=tables.Int16Atom(),
                                         shape=(0, NPMT, PMTWL_FEE),
                                         expectedrows=NEVENTS_DST,
                                         filters=tbl.filters(COMPRESSION))

            sipmrwf = h5out.create_earray(h5out.root.RD, "sipmrwf",
                                          atom=tables.Int16Atom(),
                                          shape=(0, NSIPM, SIPMWL),
                                          expectedrows=NEVENTS_DST,
                                          filters=tbl.filters(COMPRESSION))
            # LOOP
            t0 = time()
            for i in define_event_loop(CFP, NEVENTS_DST):
                # supress zeros in MCRD and rebin the ZS function in 1 mus bins
                rebin = int(units.mus/units.ns)

                # trueSiPM = wfm.zero_suppression(sipmrd_[i], 0.)
                #
                # # dict_map applies a function to the dictionary values
                # truePMT = cf.dict_map(lambda df: wfm.rebin_df(df, rebin),
                #                       wfm.zero_suppression(pmtrd_[i],
                #                       0., to_mus=int(units.ns/units.ms)))
                #
                # # store in table
                # tbl.store_wf_table(i, pmt_twf_table, truePMT)
                # tbl.store_wf_table(i, sipm_twf_table, trueSiPM)

                # simulate PMT response and return an array with RWF;BLR
                # convert to float, append to EVector

                dataPMT, blrPMT = simulate_pmt_response(i, pmtrd_)
                pmtrwf.append(dataPMT.astype(int).reshape(1, NPMT, PMTWL_FEE))
                pmtblr.append(blrPMT.astype(int).reshape(1, NPMT, PMTWL_FEE))

                # simulate SiPM response and return an array with RWF
                # convert to float, zero suppress and dump to table
                dataSiPM = simulate_sipm_response(i, sipmrd_, noise_sampler_)
                dataSiPM = wfm.to_adc(dataSiPM, sipmdf)
                dataSiPM = wfm.noise_suppression(dataSiPM, sipms_thresholds_)

                sipmrwf.append(dataSiPM.astype(int).reshape(1, NSIPM, SIPMWL))

            pmtrwf.flush()
            sipmrwf.flush()
            pmtblr.flush()

            t1 = time()
            dt = t1 - t0
            print("DIOMIRA has run over {} events in {} seconds".format(i+1,
                                                                        dt))
    print("Leaving Diomira. Safe travels!")


if __name__ == "__main__":
    # from cities import diomira
    # print(diomira)
    DIOMIRA(sys.argv)
