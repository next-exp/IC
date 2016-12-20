import sqlite3
import numpy as np
import pandas as pd
import os
from operator import itemgetter

import sys
if sys.version_info < (3,):
    from future_builtins import map


def tmap(*args):
    return tuple(map(*args))

# Run to take always the same calibration constant, etc for MC files
# 3012 was the first SiPM calibration after remapping.
runNumberForMC = 3012

def DataPMT(run_number=1e5):
    if run_number == 0:
        run_number = runNumberForMC
    dbfile = os.environ['ICDIR'] + '/Database/localdb.sqlite3'
    conn = sqlite3.connect(dbfile)
    sql = '''select map.SensorID,map.ChannelID,pos.PmtID,msk.Active,pos.X,
pos.Y,blr.coeff_c,blr.coeff_blr,gain.adc_to_pes,noise.noise_rms,sigma.sigma
from PmtMapping as map, PmtPosition as pos, PmtMask as msk, PmtBlr as blr,
PmtGain as gain, PmtNoiseRms as noise, PmtSigma as sigma
where map.SensorID=pos.SensorID and map.SensorID=msk.SensorID and
map.SensorID=blr.SensorID and map.SensorID=gain.SensorID and
map.SensorID=noise.SensorID and map.SensorID=sigma.SensorID and
map.MinRun <= {0} and (map.MaxRun >= {0} or map.MaxRun is NULL) and
pos.MinRun <= {0} and (pos.MaxRun >= {0} or pos.MaxRun is NULL) and
msk.MinRun <= {0} and (msk.MaxRun >= {0} or msk.MaxRun is NULL) and
blr.MinRun <= {0} and (blr.MaxRun >= {0} or blr.MaxRun is NULL) and
gain.MinRun <= {0} and (gain.MaxRun >= {0} or gain.MaxRun is NULL) and
sigma.MinRun <= {0} and (sigma.MaxRun >= {0} or sigma.MaxRun is NULL) and
noise.MinRun <= {0} and (noise.MaxRun >= {0} or noise.MaxRun is NULL)
order by map.SensorID;'''.format(run_number)
    data = pd.read_sql_query(sql, conn)
    conn.close()
    return data

def DataSiPM(run_number=1e5):
    if run_number == 0:
        run_number = runNumberForMC
    dbfile = os.environ['ICDIR'] + '/Database/localdb.sqlite3'
    conn = sqlite3.connect(dbfile)
    sql = '''select map.SensorID,map.ChannelID,msk.Active,pos.X,pos.Y,gain.adc_to_pes
from SipmMapping as map, SipmPosition as pos, SipmMask as msk, SipmGain as gain
where map.SensorID=pos.SensorID and map.SensorID=msk.SensorID and
map.SensorID=gain.SensorID and
map.MinRun <= {0} and (map.MaxRun >= {0} or map.MaxRun is NULL) and
pos.MinRun <= {0} and (pos.MaxRun >= {0} or pos.MaxRun is NULL) and
msk.MinRun <= {0} and (msk.MaxRun >= {0} or msk.MaxRun is NULL) and
gain.MinRun <= {0} and (gain.MaxRun >= {0} or gain.MaxRun is NULL)
order by map.SensorID;'''.format(run_number)
    data = pd.read_sql_query(sql, conn)
    conn.close()
    return data

def DetectorGeo():
    dbfile = os.environ['ICDIR'] + '/Database/localdb.sqlite3'
    conn = sqlite3.connect(dbfile)
    sql = 'select * from DetectorGeo'
    data = pd.read_sql_query(sql, conn)
    conn.close()
    return data

def SiPMNoise(run_number=1e5):
    if run_number == 0:
        run_number = runNumberForMC
    dbfile = os.environ['ICDIR'] + '/Database/localdb.sqlite3'
    conn = sqlite3.connect(dbfile)
    cursor = conn.cursor()

    sqlbaseline = '''select Energy from SipmBaseline
where MinRun <= {0} and (MaxRun >= {0} or MaxRun is NULL)
order by SensorID;'''.format(run_number)
    cursor.execute(sqlbaseline)
    baselines = np.array(tmap(itemgetter(0), cursor.fetchall()))

    sqlnoisebins = '''select Energy from SipmNoiseBins
where MinRun <= {0} and (MaxRun >= {0} or MaxRun is NULL)
order by Bin;'''.format(run_number)
    cursor.execute(sqlnoisebins)
    noise_bins = np.array(tmap(itemgetter(0), cursor.fetchall()))

    sqlnoise = '''select * from SipmNoise
where MinRun <= {0} and (MaxRun >= {0} or MaxRun is NULL)
order by SensorID;'''.format(run_number)
    cursor.execute(sqlnoise)
    data = tmap(itemgetter(slice(3,None)), cursor.fetchall())
    noise = np.array(data).reshape(1792, 300)

    return noise, noise_bins, baselines
