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
import invisible_cities.sierpe.blr as blr
import invisible_cities.core.peak_functions_c as pf
from invisible_cities.database import load_db



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
