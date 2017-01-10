"""Functions to find peaks, S12 selection etc.
JJGC and GML December 2016
"""
from __future__ import print_function, division, absolute_import

import math
import numpy as np
import pandas as pd
from time import time
import tables as tb
import matplotlib.pyplot as plt

import invisible_cities.core.system_of_units as units
import invisible_cities.sierpe.BLR as blr
import invisible_cities.core.peak_functions_c as pf
from invisible_cities.database import load_db


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
