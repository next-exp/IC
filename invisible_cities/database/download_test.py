import os

import sqlite3
import pymysql
pymysql.install_as_MySQLdb()

from pytest import mark

from . import download as db

@mark.parametrize('dbname', 'DEMOPPDB NEWDB NEXT100DB'.split())
def test_create_table_sqlite(dbname, output_tmpdir):
    dbfile = os.path.join(output_tmpdir, 'db.sqlite3')

    if os.path.isfile(dbfile):
        os.remove(dbfile)

    dbname = 'NEXT100DB'
    table = 'PmtBlr'

    connSqlite = sqlite3.connect(dbfile)
    connMySql  = pymysql.connect(host="neutrinos1.ific.uv.es",
                                user='nextreader',passwd='readonly', db=dbname)

    cursorMySql  = connMySql .cursor()
    cursorSqlite = connSqlite.cursor()

    for table in db.tables:
        db.create_table_sqlite(cursorSqlite, cursorMySql, table)
