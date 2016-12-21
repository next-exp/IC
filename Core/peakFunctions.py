"""Functions to find peaks, S12 selection etc.
JJGC and GML December 2016
"""
from __future__ import print_function, division, absolute_import

import math
import numpy as np
import pandas as pd
import Core.system_of_units as units
import ICython.Sierpe.BLR as blr
import ICython.Core.peakFunctions as pf
from Database import loadDB
import matplotlib.pyplot as plt
from time import time
import tables as tb



def pmt_sum(CWF, adc_to_pes):
    """
    input: A CWF list or array
           a vector with the adc_to_pes values (must be positive)
    returns: the sum of CWF, in pes

    """

    NPMT = len(CWF)
    NWF = len(CWF[0])

    csum = np.zeros(NWF, dtype=np.double)
    for j in range(NPMT):
        csum += CWF[j] * 1 / adc_to_pes[j]
    return csum


def wfdf(time,energy_pes):
    """Take two vectors (time, energy) and return a data frame
    representing a waveform."""
    swf = {}
    swf['time_ns'] = time / units.ns
    swf['ene_pes'] = energy_pes
    return pd.DataFrame(swf)


def wf_thr(wf, threshold=0):
    """Return a zero supressed waveform (more generally, the vaules of wf
    above threshold).
    """
    return wf.loc[lambda df: df.ene_pes.values > threshold, :]


def find_peaks(wfzs, stride=4, lmin=8):
    """Find peaks.

    Do not interrupt the peak if next sample comes within stride
    accept the peak only if larger than lmin samples
    """
    T = wfzs['time_mus'].values
    P = wfzs['ene_pes'].values
    I = wfzs.index.values

    S12 = {}
    pulse_on = 1
    j=0

    S12[0] = []
    S12[0].append([T[0], P[0], I[0]])

    for i in range(1, len(wfzs)) :
        if wfzs.index[i]-stride > wfzs.index[i-1]:  #new s12
            j+=1
            S12[j] = []
            S12[j].append([T[i], P[i], I[i]])
        else:
            S12[j].append([T[i], P[i], I[i]])

    S12L=[]
    for i in S12.keys():
        if len(S12[i]) > lmin:
            S12L.append(pd.DataFrame(S12[i], columns=['time_mus','ene_pes','index']))
    return S12L


def find_S12(wfzs, tmin=0*units.mus, tmax=1200*units.mus,
             stride=4, lmin=8, lmax=1e+6):
    """Find S1/S2 peaks.

    input: a zero supressed wf
    returns a list of waveform data frames
    do not interrupt the peak if next sample comes within stride
    accept the peak only if within [lmin, lmax)
    accept the peak only if within [tmin, tmax)
    """

    T = wfzs['time_ns'].values
    P = wfzs['ene_pes'].values

    S12 = {}
    pulse_on = 1
    j=0

    S12[0] = []
    S12[0].append([T[0],P[0]])

    for i in range(1, len(wfzs)):

        if T[i] > tmax:
            break

        if T[i] < tmin:
            continue

        if wfzs.index[i] - stride > wfzs.index[i-1]:  #new s12
            j += 1
            S12[j] = []
            S12[j].append([T[i], P[i]])
        else:
            S12[j].append([T[i], P[i]])

    S12L=[]
    for i in S12.keys():
        if len(S12[i]) >= lmin and len(S12[i]) < lmax:
            S12L.append(pd.DataFrame(S12[i], columns=['time_ns','ene_pes']))
    return S12L

def sipm_S2(dSIPM,S2, thr=5*units.pes):
    """Given a vector with SIPMs (energies above threshold), return a list
    of np arrays. Each element of the list is the S2 window in the
    SiPM (if not zero).
    """

    i0,i1 = index_from_S2(S2)
    dim = int(i1 - i0)
    SIPML = []
    for i in dSIPM.keys():
        sipm = dSIPM[i][1]
        psum = np.sum(sipm[i0:i1])
        if psum > thr:
            e = np.zeros(dim, dtype=np.double)
            e[:] = sipm[i0:i1]
            SIPML.append([dSIPM[i][0], e])
    return SIPML


def dict_to_df_S12(S12):
    """Take an S12 dictionary and return a list of DF."""
    S12L = []
    print('number of peaks = {}'.format(len(S12)))
    for i in S12.keys():
        s12df = pd.DataFrame(S12[i], columns=['time_ns','ene_pes'])
        print('S12 number = {}, samples = {} sum in pes ={}'.\
          format(i, len(s12df), np.sum(s12df.ene_pes.values)))
        S12L.append(s12df)
    return S12L

def scan_S12(S12):
    """Print and plot the peaks of input S12.

    S12 is a dictionary
    S12[i] for i in keys() are the S12 peaks
    """
    print('number of peaks = {}'.format(len(S12)))
    for i in S12.keys():
        print('S12 number = {}, samples = {} sum in pes ={}'.\
          format(i, len(S12[i][0]), np.sum(S12[i][1])))
        plt.plot(S12[i][0], S12[i][1])
        plt.show()
        raw_input('hit return')


def index_from_S2(S2):
    """Return the indexes defining the vector."""
    T = S2[0] / units.mus
    #print(T[0], T[-1])
    return int(T[0]), int(T[-1])



def sipm_S2_dict(SIPM, S2d, thr=5 * units.pes):
    """Given a vector with SIPMs (energies above threshold), and a
    dictionary of S2s, S2d, returns a dictionary of SiPMs-S2.  Each
    index of the dictionary correspond to one S2 and is a list of np
    arrays. Each element of the list is the S2 window in the SiPM (if
    not zero)
    """
    SiPMd = {}
    for i in S2d.keys():
        S2 = S2d[i]
        SiPMd[i] = sipm_S2(SIPM, S2, thr=thr)
    return SiPMd

def scan_S12L(S12L):
    """Print and plot the peaks of input list S12L S12L is a list of data
    frames.
    """
    print('number of peaks = {}'.format(len(S12L)))
    for i, s12df in enumerate(S12L):
        print('S12 number = {}, samples = {} sum in pes ={}'.\
          format(i, len(s12df), np.sum(s12df.ene_pes.values)))
        plt.plot(s12df.time_ns.values, s12df.ene_pes)
        plt.show()
        raw_input('hit return')


class S12Finder:
    """Driver class to find S12."""
    def __init__(self, run_number, n_baseline=28000, n_MAU=200,
                 thr_trigger=5, wfm_length=48000, tstep = 25):
        """
        Inits the machine
        """
        DataPMT = loadDB.DataPMT(run_number)
        self.adc_to_pes = abs(DataPMT.adc_to_pes.values).astype(np.double)
        self.coeff_c = DataPMT.coeff_c.values.astype(np.double)
        self.coeff_blr = DataPMT.coeff_blr.values.astype(np.double)

        self.n_MAU = n_MAU
        self.n_baseline = n_baseline
        self.thr_trigger = thr_trigger
        self.signal_t = np.arange(0, wfm_length * tstep, tstep)

        self.setFiles = False
        self.setS1 = False
        self.setS2 = False

        self.plot_csum = False
        self.plot_s1 = False
        self.plot_s2 = False

        self.nprint = 1000000

        # Dictionary of S1 and S2
        # each entry contains a list of S1 and S2 df

        self.dS1 = {}
        self.dS2 = {}

    def set_plot(self, plot_csum=False, plot_s1=False, plot_s2=False):
        self.plot_csum = plot_csum
        self.plot_s1 = plot_s1
        self.plot_s2 = plot_s2

    def set_print(self, nprint=10):
        self.nprint = nprint

    def set_files(self,path, input_files):
        """Set the input files."""
        self.path = path
        self.input_files = input_files
        self.setFiles = True

    def set_s1(self, tmin=0*units.mus, tmax=590*units.mus, stride=4, lmin=4, lmax=16):
        self.tmin_s1 = tmin
        self.tmax_s1 = tmax
        self.stride_s1 = stride
        self.lmin_s1 = lmin
        self.lmax_s1 = lmax
        self.setS1 = True

    def set_s2(self, tmin=590*units.mus, tmax=620*units.mus, stride=40, lmin=100, lmax=1000000):
        self.tmin_s2 = tmin
        self.tmax_s2 = tmax
        self.stride_s2 = stride
        self.lmin_s2 = lmin
        self.lmax_s2 = lmax
        self.setS2 = True

    def get_dS1(self):
        if len(self.dS1) == 0:
            print('S1 dictionary is empty')
            return 0
        else:
            return self.dS1

    def get_dS2(self):
        if len(self.dS2) == 0:
            print('S2 dictionary is empty')
            return 0
        else:
            return self.dS2

    def find_s12(self, nmax, thr_s12=1.0*units.pes):
        """Run the machine."""
        n_events_tot = 0

        if self.setFiles == False:
            raise IOError('must set files before running')
        if self.setS1 == False:
            raise IOError('must set S1 parameters before running')
        if self.setS2 == False:
            raise IOError('must set S2 parameters before running')
        if self.path =='':
            raise IOError('path is empty')
        if len(self.input_files) == 0:
            raise IOError('input file list is empty')


        t0 = time()
        print('t0 = {} s'.format(t0))
        for ffile in self.input_files:

            print("Opening", ffile, end="... ")
            filename = self.path + ffile
            #sys.stdout.flush()

            try:
                with tb.open_file(filename, "r+") as h5in:

                    pmtrwf = h5in.root.RD.pmtrwf
                    NEVT = pmtrwf.shape[0]

                    for evt in range(NEVT):
                        # deconvolve
                        CWF = blr.deconv_pmt(pmtrwf[evt], self.coeff_c, self.coeff_blr,
                                             n_baseline=self.n_baseline,
                                             thr_trigger=self.thr_trigger)

                        # calibrated PMT sum
                        csum = pf.calibrated_pmt_sum(CWF, self.adc_to_pes,
                                                     n_MAU=self.n_MAU,
                                                     thr_MAU=self.thr_trigger)
                        if self.plot_csum:
                            plt.plot(csum)
                            plt.show()
                            raw_input('->')


                        # Supress samples below threshold (in pes)
                        wfzs_ene, wfzs_indx = pf.wfzs(csum, threshold=thr_s12)

                        # find S1 and S2
                        S1 = pf.find_S12(wfzs_ene, wfzs_indx,
                                         tmin=self.tmin_s1,
                                         tmax=self.tmax_s1,
                                         stride=self.stride_s1)

                        s1df = s12_df(S1, lmin=self.lmin_s1, lmax=self.lmax_s1)

                        S2 = pf.find_S12(wfzs_ene, wfzs_indx,
                                         tmin=self.tmin_s2,
                                         tmax=self.tmax_s2,
                                         stride=self.stride_s2)
                        s2df = s12_df(S2, lmin=self.lmin_s2, lmax=self.lmax_s2)

                        self.dS1[n_events_tot] = s1df
                        self.dS2[n_events_tot] = s2df

                        if self.plot_s1:
                            scan_S12(s1df)
                        if self.plot_s2:
                            scan_S12(s2df)

                        n_events_tot +=1
                        if n_events_tot%self.nprint == 0:
                            print('event in file = {}, total = {}'.format(evt, n_events_tot))

                        if n_events_tot > nmax:
                            print('reached maximum number of events (={})'.format(nmax))
                            self.plot_csum = False
                            self.plot_s1 = False
                            self.plot_s2 = False
                            break


            except:
                print('error')
                raise


        t1 = time()
        print('t1 = {} s'.format(t1))
        dt = t1 - t0

        print("S12Finder has run over {} events in {} seconds".format(n_events_tot+1, dt))
