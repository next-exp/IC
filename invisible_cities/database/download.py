import sqlite3
import sys
import pymysql
pymysql.install_as_MySQLdb()
import os
import re
from os import path
from datetime import datetime

# Absolute imports to allow usage as standalone program:
# python invisible_cities/database/download.py
from invisible_cities.database.db_connection import connect_sqlite
from invisible_cities.database.db_connection import connect_mysql
from invisible_cities.database.db_connection import connect_dolt_mysql


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


def loadDB(dbname : str, tables : list):
    print("Cloning database {}".format(dbname))
    dbfile = path.join(os.environ['ICDIR'], 'database/localdb.'+dbname+'.sqlite3')
    try:
        os.remove(dbfile)
    except:
        pass

    conn_sqlite, cursor_sqlite = connect_sqlite(dbfile)
    conn_mysql , cursor_mysql  = connect_mysql (dbname)

    for table in tables:
        print("Downloading table {}".format(table))

        # Create table
        create_table_sqlite(cursor_sqlite, cursor_mysql, table)

        # Copy data
        copy_all_rows(conn_sqlite, cursor_sqlite, cursor_mysql, table)

def write_db_version(dbname: str):
    dbfile = path.join(os.environ['ICDIR'], 'database/localdb.'+dbname+'.sqlite3')
    try:
        conn_mysql , cursor_mysql  = connect_dolt_mysql(dbname)
    except pymysql.err.OperationalError:
        print(f"DB versioning not implemented for database {dbname}")
        return
    conn_sqlite, cursor_sqlite = connect_sqlite(dbfile)
    sql = 'select commit_hash, date from dolt_diff order by date desc limit 1'
    cursor_mysql.execute(sql)
    data = cursor_mysql.fetchall()
    if len(data) == 1:
        db_version = data[0][0]
        date = data[0][1]
        timestamp = datetime.timestamp(date)
        sql_table = "create table db_version(version VARCHAR(20), date timestamp null)"
        cursor_sqlite.execute(sql_table)
        sql_value = f'insert into db_version (version, date) values ("{db_version}", {timestamp})'
        cursor_sqlite.execute(sql_value)
        conn_sqlite.commit()


dbnames        = ('NEWDB', 'DEMOPPDB', 'NEXT100DB', 'Flex100DB')
common_tables  = ('DetectorGeo','PmtBlr','ChannelGain','ChannelMapping','ChannelMask',
                  'PmtNoiseRms','ChannelPosition','SipmBaseline', 'SipmNoisePDF',
                  'PMTFEMapping', 'PMTFELowFrequencyNoise')
extended       = dict(NEXT100DB = ("Activity", "Efficiency"))

table_dict = dict.fromkeys(dbnames, common_tables)
for dbname, extra in extended.items():
    table_dict[dbname] += extra

if __name__ == '__main__':
    if len(sys.argv) > 1:
        dbname = sys.argv[1]
        loadDB(dbname, table_dict[dbname])
        write_db_version(dbname)
    else:
        for dbname, tables in table_dict.items():
            loadDB(dbname, tables)
            write_db_version(dbname)
