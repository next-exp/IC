import sqlite3
import sys
import pymysql
import pymysql as MySQLdb
pymysql.install_as_MySQLdb()
import os
from os import path


def loadDB(dbname='NEWDB'):
    dbfile = path.join(os.environ['ICDIR'], 'database/localdb.sqlite3')
    try:
        os.remove(dbfile)
    except:
        pass

    connSql3 = sqlite3.connect(dbfile)
    cursorSql3 = connSql3.cursor()


    connMySql = MySQLdb.connect(host="neutrinos1.ific.uv.es", user='nextreader',passwd='readonly', db=dbname)
    cursorMySql = connMySql.cursor()

    # Create tables
    cursorSql3.execute('''CREATE TABLE IF NOT EXISTS `DetectorGeo` (
  `XMIN` float NOT NULL
,  `XMAX` float NOT NULL
,  `YMIN` float NOT NULL
,  `YMAX` float NOT NULL
,  `ZMIN` float NOT NULL
,  `ZMAX` float NOT NULL
,  `RMAX` float NOT NULL
);''')

    cursorSql3.execute('''CREATE TABLE IF NOT EXISTS `ChannelGain` (
  `MinRun` integer NOT NULL
,  `MaxRun` integer NOT NULL
,  `SensorID` integer NOT NULL
,  `Centroid` float NOT NULL
,  `ErrorCentroid` float NOT NULL
,  `Sigma` float NOT NULL
,  `ErrorSigma` float NOT NULL
);''')

    cursorSql3.execute('''CREATE TABLE IF NOT EXISTS `ChannelMask` (
  `MinRun` integer NOT NULL
,  `MaxRun` integer NOT NULL
,  `SensorID` integer NOT NULL
);''')

    cursorSql3.execute('''CREATE TABLE IF NOT EXISTS `ChannelMapping` (
  `MinRun` integer NOT NULL
,  `MaxRun` integer NOT NULL
,  `ElecID` integer NOT NULL
,  `SensorID` integer NOT NULL
);''')

    cursorSql3.execute('''CREATE TABLE IF NOT EXISTS `ChannelPosition` (
  `MinRun` integer NOT NULL
,  `MaxRun` integer NOT NULL
,  `SensorID` integer NOT NULL
,  `Label` varchar(20) NOT NULL
,  `Type` varchar(20) NOT NULL
,  `X` float NOT NULL
,  `Y` float NOT NULL
);''')

    cursorSql3.execute('''CREATE TABLE IF NOT EXISTS `PmtBlr` (
  `MinRun` integer NOT NULL
,  `MaxRun` integer DEFAULT NULL
,  `ElecID` integer NOT NULL
,  `coeff_c` double NOT NULL
,  `coeff_blr` double NOT NULL
);''')

    cursorSql3.execute('''CREATE TABLE IF NOT EXISTS `PmtNoiseRms` (
  `MinRun` integer NOT NULL
,  `MaxRun` integer DEFAULT NULL
,  `ElecID` integer NOT NULL
,  `noise_rms` double NOT NULL
);''')


    cursorSql3.execute('''CREATE TABLE IF NOT EXISTS `SipmBaseline` (
  `MinRun` integer NOT NULL
,  `MaxRun` integer DEFAULT NULL
,  `SensorID` integer NOT NULL
,  `Energy` float NOT NULL
);''')

    cursorSql3.execute('''CREATE TABLE IF NOT EXISTS `SipmNoisePDF` (
  `MinRun` integer NOT NULL
,  `MaxRun` integer DEFAULT NULL
,  `SensorID` integer NOT NULL
,  `BinEnergyPes` float NOT NULL
,  `Probability` float NOT NULL
);''')

    cursorSql3.execute('''CREATE TABLE IF NOT EXISTS `PMTFEMapping` (
    `MinRun` integer NOT NULL
,  `MaxRun` integer NOT NULL
,  `SensorID` integer NOT NULL
,  `FEBox` integer NOT NULL
);''')

    cursorSql3.execute('''CREATE TABLE IF NOT EXISTS `PMTFELowFrequencyNoise` (
    `MinRun` integer NOT NULL
,  `MaxRun` integer NOT NULL
,  `Frequency` float NOT NULL
,  `FE0Magnitude` float NOT NULL
,  `FE1Magnitude` float NOT NULL
,  `FE2Magnitude` float NOT NULL
);''')


    tables = ['DetectorGeo','PmtBlr','ChannelGain','ChannelMapping','ChannelMask',
              'PmtNoiseRms','ChannelPosition','SipmBaseline', 'SipmNoisePDF',
              'PMTFEMapping', 'PMTFELowFrequencyNoise']


    # Copy all tables
    for table in tables:
        # Get all data
        cursorMySql.execute('SELECT * from {0}'.format(table))
        data = cursorMySql.fetchall()

        # Insert all rows
        fields = '?'
        nfields = len(data[0])
        fields += (nfields-1) * ',?'
        cursorSql3.executemany('INSERT INTO {0} VALUES({1})'.format(table,fields),data)
        connSql3.commit()


if __name__ == '__main__':
    dbname = 'NEWDB'
    if len(sys.argv) > 1:
        dbname = sys.argv[1]
    loadDB(dbname)
