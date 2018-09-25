import sqlite3
import numpy as np
import pandas as pd
import os
from operator  import itemgetter
from functools import lru_cache

DATABASE_LOCATION =  os.environ['ICTDIR'] + '/invisible_cities/database/localdb.sqlite3'


def tmap(*args):
    return tuple(map(*args))

# Run to take always the same calibration constant, etc for MC files
# 3012 was the first SiPM calibration after remapping.
runNumberForMC = 3012

@lru_cache(maxsize=10)
def DataPMT(run_number=1e5, db_file=DATABASE_LOCATION):
    if run_number == 0:
        run_number = runNumberForMC

    conn = sqlite3.connect(db_file)

    sql = '''select pos.SensorID, map.ElecID "ChannelID", Label "PmtID",
case when msk.SensorID is NULL then 1 else 0 end "Active",
X, Y, coeff_blr, coeff_c, abs(Centroid) "adc_to_pes", noise_rms, Sigma
from ChannelPosition as pos INNER JOIN ChannelMapping
as map ON pos.SensorID = map.SensorID LEFT JOIN
(select * from PmtNoiseRms where MinRun <= {0} and (MaxRun >= {0} or MaxRun is NULL))
as noise on map.ElecID = noise.ElecID LEFT JOIN
(select * from ChannelMask where MinRun <= {0} and {0} <= MaxRun)
as msk ON pos.SensorID = msk.SensorID LEFT JOIN
(select * from ChannelGain where  MinRun <= {0} and {0} <= MaxRun)
as gain ON pos.SensorID = gain.SensorID LEFT JOIN
(select * from PmtBlr where MinRun <= {0} and (MaxRun >= {0} or MaxRun is NULL))
as blr ON map.ElecID = blr.ElecID
where pos.SensorID < 100
and pos.MinRun <= {0} and {0} <= pos.MaxRun
and map.MinRun <= {0} and {0} <= map.MaxRun
and pos.Label LIKE 'PMT%'
order by Active desc, pos.SensorID
'''.format(abs(run_number))
    data = pd.read_sql_query(sql, conn)
    data.fillna(0, inplace=True)
    conn.close()
    return data

@lru_cache(maxsize=10)
def DataSiPM(run_number=1e5, db_file=DATABASE_LOCATION):
    if run_number == 0:
        run_number = runNumberForMC

    conn = sqlite3.connect(db_file)

    sql='''select pos.SensorID, map.ElecID "ChannelID",
case when msk.SensorID is NULL then 1 else 0 end "Active",
X, Y, Centroid "adc_to_pes", Sigma
from ChannelPosition as pos INNER JOIN ChannelGain as gain
ON pos.SensorID = gain.SensorID INNER JOIN ChannelMapping as map
ON pos.SensorID = map.SensorID LEFT JOIN
(select * from ChannelMask where MinRun <= {0} and {0} <= MaxRun) as msk
ON pos.SensorID = msk.SensorID
where pos.SensorID > 100
and pos.MinRun <= {0} and {0} <= pos.MaxRun
and gain.MinRun <= {0} and {0} <= gain.MaxRun
and map.MinRun <= {0} and {0} <= map.MaxRun
order by pos.SensorID'''.format(abs(run_number))
    data = pd.read_sql_query(sql, conn)
    conn.close()

    ## Add default value to Sigma for runs without measurement
    if not data.Sigma.values.any():
        data.Sigma = 2.24

    return data

@lru_cache(maxsize=10)
def DetectorGeo(db_file=DATABASE_LOCATION):
    conn = sqlite3.connect(db_file)
    sql = 'select * from DetectorGeo'
    data = pd.read_sql_query(sql, conn)
    conn.close()
    return data

@lru_cache(maxsize=10)
def SiPMNoise(run_number=1e5, db_file=DATABASE_LOCATION):
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()

    sqlbaseline = '''select Energy from SipmBaseline
where MinRun <= {0} and (MaxRun >= {0} or MaxRun is NULL)
order by SensorID;'''.format(abs(run_number))
    cursor.execute(sqlbaseline)
    baselines = np.array(tmap(itemgetter(0), cursor.fetchall()))
    nsipms = baselines.shape[0]

    sqlnoisebins = '''select distinct(BinEnergyPes) from SipmNoisePDF
where MinRun <= {0} and (MaxRun >= {0} or MaxRun is NULL)
order by BinEnergyPes;'''.format(abs(run_number))
    cursor.execute(sqlnoisebins)
    noise_bins = np.array(tmap(itemgetter(0), cursor.fetchall()))
    nbins = noise_bins.shape[0]

    sqlnoise = '''select Probability from SipmNoisePDF
where MinRun <= {0} and (MaxRun >= {0} or MaxRun is NULL)
order by SensorID, BinEnergyPes;'''.format(abs(run_number))
    cursor.execute(sqlnoise)
    data = tmap(itemgetter(0), cursor.fetchall())
    noise = np.array(data).reshape(nsipms, nbins)

    return noise, noise_bins, baselines


@lru_cache(maxsize=10)
def PMTLowFrequencyNoise(run_number=1e5, dbfile=DATABASE_LOCATION):
    conn = sqlite3.connect(dbfile)
    cursor = conn.cursor()

    sqlmapping = '''select SensorID, FEBox from PMTFEMapping
    where MinRun <= {0} and (MaxRun >= {0} or MaxRun is NULL)
    order by SensorID;'''.format(abs(run_number))
    mapping = pd.read_sql_query(sqlmapping, conn)

    ## Now get the frequencies and magnitudes (and ?) for each box
    sqlmagnitudes = '''select Frequency, FE0Magnitude, FE1Magnitude, FE2Magnitude
    from PMTFELowFrequencyNoise where MinRun <= {0}
    and (MaxRun >= {0} or MaxRun is NULL)'''.format(abs(run_number))
    frequencies = pd.read_sql_query(sqlmagnitudes, conn)

    return mapping, frequencies.values
