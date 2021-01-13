import sqlite3
import pymysql
pymysql.install_as_MySQLdb()


def connect_sqlite(dbfile):
    conn_sqlite   = sqlite3.connect(dbfile)
    cursor_sqlite = conn_sqlite.cursor()
    return conn_sqlite, cursor_sqlite


def connect_mysql(dbname):
    conn_mysql  = pymysql.connect(host="neutrinos1.ific.uv.es",
                                  user='nextreader',passwd='readonly', db=dbname)
    cursor_mysql  = conn_mysql .cursor()
    return connect_mysql, cursor_mysql

