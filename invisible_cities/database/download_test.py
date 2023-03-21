import os

import sqlite3
import pymysql
pymysql.install_as_MySQLdb()

from pytest import mark

from . import download as db

@mark.skip(reason='server timeouts cause too many spurious test failures')
@mark.parametrize('dbname', 'DEMOPPDB NEWDB NEXT100DB Flex100DB'.split())
def test_create_table_sqlite(dbname, output_tmpdir):
    dbfile = os.path.join(output_tmpdir, 'db.sqlite3')

    if os.path.isfile(dbfile):
        os.remove(dbfile)

    dbname = 'NEXT100DB'
    table = 'PmtBlr'

    connSqlite = sqlite3.connect(dbfile)
    connMySql  = pymysql.connect(host="next.ific.uv.es",
                                 user='nextreader',passwd='readonly', db=dbname)

    cursorMySql  = connMySql .cursor()
    cursorSqlite = connSqlite.cursor()

    for table in db.tables:
        db.create_table_sqlite(cursorSqlite, cursorMySql, table)


@mark.parametrize('dbname', db.dbnames)
def test_table_assignment(dbname):
    for name in db.common_tables:
        assert name in db.table_dict[dbname]

    for name in db.extended.get(dbname, ()):
        assert name in db.table_dict[dbname]


@mark.parametrize('dbname', db.dbnames)
def test_tables_exist(dbname):
    connMySql  = pymysql.connect(host="next.ific.uv.es",
                                 user='nextreader',passwd='readonly', db=dbname)

    cursor    = connMySql.cursor()
    cursor.execute("Show tables;")
    available = cursor.fetchall()

    for name in db.table_dict[dbname]:
        assert (name,) in available
        

