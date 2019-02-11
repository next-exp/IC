import sqlite3
import sys
import pymysql
pymysql.install_as_MySQLdb()
import os
import re
from os import path


tables = ['DetectorGeo','PmtBlr','ChannelGain','ChannelMapping','ChannelMask',
          'PmtNoiseRms','ChannelPosition','SipmBaseline', 'SipmNoisePDF',
          'PMTFEMapping', 'PMTFELowFrequencyNoise']


def create_table_sqlite(cursorSqlite, cursorMySql, table):
    cursorMySql.execute('show create table {}'.format(table))
    data = cursorMySql.fetchone()
    sql  = data[1]

    # show create table will include comments for each row
    # Example: `MinRun` int(11) NOT NULL COMMENT 'Minimum run number for which valid'
    # that is not compatible with sqlite3, so we remove it.
    sql = re.sub(r" COMMENT\s+\'.*\'", "", sql)

    # it will also include: ENGINE=MyISAM DEFAULT CHARSET=latin1
    # this is not compatible either, so we remove it.
    sql = re.sub(r"\s*ENGINE.*", "", sql)

    # some tables may have KEYs defined: KEY `ElecID` (`SensorID`)
    # this syntax is different, so we remove it too.
    # This happens for instance in table ChannelGain
    sql = re.sub(r",\s*\n\s*KEY.*[\n,]", "", sql)

    cursorSqlite.execute(sql)


def copy_all_rows(connSqlite, cursorSqlite, cursorMySql, table):
    # Get all data
    cursorMySql.execute('SELECT * from {0}'.format(table))
    data = cursorMySql.fetchall()

    # Insert all rows
    fields = '?'
    try:
        nfields = len(data[0])
        fields += (nfields-1) * ',?'
        cursorSqlite.executemany('INSERT INTO {0} VALUES({1})'.format(table,fields),data)
        connSqlite.commit()
    except IndexError:
        print('Table ' +table+' is empty.')


def loadDB(dbname : str):
    print("Cloning database {}".format(dbname))
    dbfile = path.join(os.environ['ICDIR'], 'database/localdb.'+dbname+'.sqlite3')
    try:
        os.remove(dbfile)
    except:
        pass

    connSqlite = sqlite3.connect(dbfile)
    connMySql  = pymysql.connect(host="neutrinos1.ific.uv.es",
                                user='nextreader',passwd='readonly', db=dbname)

    cursorMySql  = connMySql .cursor()
    cursorSqlite = connSqlite.cursor()

    for table in tables:
        print("Downloading table {}".format(table))

        # Create table
        create_table_sqlite(cursorSqlite, cursorMySql, table)

        # Copy data
        copy_all_rows(connSqlite, cursorSqlite, cursorMySql, table)


if __name__ == '__main__':
    dbname = ['NEWDB', 'DEMOPPDB', 'NEXT100DB']
    if len(sys.argv) > 1:
        dbname = sys.argv[1]
        loadDB(dbname)
    else:
        for name in dbname:
            loadDB(name)


