import sqlite3
import numpy as np
import pandas as pd
import os
from operator import itemgetter

import sys
if sys.version_info < (3,):
    from future_builtins import map


DATABASE_LOCATION = '/invisible_cities/database/localdb.sqlite3'

def tmap(*args):
    return tuple(map(*args))

# Run to take always the same calibration constant, etc for MC files
# 3012 was the first SiPM calibration after remapping.
runNumberForMC = 3012

def get_min_run_values(db, run_number, sensors):
    ''' Due to ART database design MaxRun is not being used currently,
    so we need to get the exact MinRun value for a given run number to
    get only one row per sensor...
    db is sqlite3 connection and sensors pmt or sipm.'''

    if sensors == 'pmt':  bound = '<' #pmt id's are < 100
    if sensors == 'sipm': bound = '>' #sipms id's are > 100

    sql = '''select Max(MinRun) from ChannelGain
where SensorID {} 100 and MinRun < {}'''.format(bound, run_number)
    cursor = db.execute(sql)
    minrun_gain = cursor.fetchone()[0]

    sql = '''select Max(MinRun) from ChannelPosition
where SensorID {} 100 and MinRun < {}'''.format(bound, run_number)
    cursor = db.execute(sql)
    minrun_position = cursor.fetchone()[0]

    sql = '''select Max(MinRun) from ChannelMapping
where SensorID {} 100 and MinRun < {}'''.format(bound, run_number)
    cursor = db.execute(sql)
    minrun_map = cursor.fetchone()[0]

    return minrun_gain, minrun_position, minrun_map


def DataPMT(run_number=1e5):
    if run_number == 0:
        run_number = runNumberForMC
    dbfile = os.environ['ICTDIR'] + DATABASE_LOCATION
    conn = sqlite3.connect(dbfile)

    minrun_gain, minrun_position, minrun_map = \
            get_min_run_values(conn, run_number, 'pmt')

    sql = '''select pos.SensorID, map.ElecID "ChannelID", Label "PmtID",
case when msk.SensorID is NULL then 1 else 0 end "Active",
X, Y, coeff_blr, coeff_c, abs(Centroid) "adc_to_pes", noise_rms, Sigma
from ChannelPosition as pos INNER JOIN ChannelMapping
as map ON pos.SensorID = map.SensorID LEFT JOIN
(select * from PmtNoiseRms where MinRun < {3} and (MaxRun >= {3} or MaxRun is NULL))
as noise on map.ElecID = noise.ElecID LEFT JOIN
(select * from ChannelMask where MinRun < {3} and {3} < MaxRun)
as msk ON pos.SensorID = msk.SensorID LEFT JOIN
(select * from ChannelGain where MinRun={1})
as gain ON pos.SensorID = gain.SensorID LEFT JOIN
(select * from PmtBlr where MinRun < {3} and (MaxRun >= {3} or MaxRun is NULL))
as blr ON map.ElecID = blr.ElecID
where pos.SensorID < 100 and pos.MinRun={0} and map.MinRun={2}
order by Active desc, pos.SensorID'''\
    .format(minrun_position, minrun_gain, minrun_map, run_number)
    data = pd.read_sql_query(sql, conn)
    data.fillna(0, inplace=True)
    conn.close()
    return data

def DataSiPM(run_number=1e5):
    if run_number == 0:
        run_number = runNumberForMC
    dbfile = os.environ['ICTDIR'] + DATABASE_LOCATION
    conn = sqlite3.connect(dbfile)

    minrun_gain, minrun_position, minrun_map = \
            get_min_run_values(conn, run_number, 'sipm')

    sql = '''select pos.SensorID, map.ElecID "ChannelID",
case when msk.SensorID is NULL then 1 else 0 end "Active",
X, Y, Centroid "adc_to_pes"
from ChannelPosition as pos INNER JOIN ChannelGain as gain
ON pos.SensorID = gain.SensorID INNER JOIN ChannelMapping as map
ON pos.SensorID = map.SensorID LEFT JOIN
(select * from ChannelMask where MinRun < {3} and {3} < MaxRun) as msk
ON pos.SensorID = msk.SensorID
where pos.SensorID > 100 and pos.MinRun={0} and gain.MinRun={1}
and map.MinRun={2} order by pos.SensorID'''\
    .format(minrun_position, minrun_gain, minrun_map, run_number)
    data = pd.read_sql_query(sql, conn)
    conn.close()
    return data

def DetectorGeo():
    dbfile = os.environ['ICTDIR'] + DATABASE_LOCATION
    conn = sqlite3.connect(dbfile)
    sql = 'select * from DetectorGeo'
    data = pd.read_sql_query(sql, conn)
    conn.close()
    return data

def SiPMNoise(run_number=1e5):
    if run_number == 0:
        run_number = runNumberForMC
    dbfile = os.environ['ICTDIR'] + DATABASE_LOCATION
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
