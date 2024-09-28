# from DBUtils.PooledDB import PooledDB
import pymysql
import os
dbinfo = {
    'host': 'localhost',
    'port': 3306,
    'user': 'root',
    'password': 'QJCqjc777',
    'database': 'ava'
}



class mysql:
    def __init__(self):
        self.reconnect()

    def reconnect(self):
        self.db = pymysql.connect(host=dbinfo['host'],
                                  user=dbinfo['user'],
                                  password=dbinfo['password'],
                                  database=dbinfo['database'])
        self.cursor = self.db.cursor()

    def query(self, sql):
        try:
            self.cursor.execute(sql)
            return self.cursor.fetchall()
        except pymysql.err.OperationalError as e:
            # 链接超时断开
            self.reconnect()
            return self.query(sql)

    def close(self):
        self.db.close()
