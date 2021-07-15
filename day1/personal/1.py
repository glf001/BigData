#!/usr/bin/env python
# -*-coding:utf-8 -*-
# author:GLF time:2020.3.24

import pymysql.cursors
import sys
import importlib
import pandas as pd
importlib.reload(sys)
import pymysql
import os

def export(table_name,outputpath):
    # lst = os.listdir(os.getcwd())
    # for c in lst:
    #     if os.path.isfile(c) and c.endswith('.py'):  # 去掉AllTest.py文件
    #         print(c)
    #         os.system(os.path.join(os.getcwd(), c))
    pd.set_option('display.max_columns', None)
    pd.set_option('display.unicode.ambiguous_as_wide', True)
    pd.set_option('display.unicode.east_asian_width', True)

    conn = pymysql.connect(
        host='127.0.0.1',
        port=3306,
        user='root',
        passwd='root',
        db='xushui',
        charset='utf8')

    cur = conn.cursor()
    sql = "select * from dawu where his名称 like '%一次性注射器%'"

    df = pd.read_sql(sql,con= conn)
    df.insert(0, "数量合", None)
    df.insert(0, "金额合", None)
    df.insert(0, "条数", None)
    df["数量"] = df["数量"].astype('float')
    df["金额"] = df["金额"].astype('float')
    df["数量合"] = df["数量"].sum()
    df["金额合"] = df["金额"].sum()
    df["条数"] = df["HIS名称"].count()
    df.to_excel('C:\\Users\\glf\\Desktop\\001.xlsx',index= False)
    count = cur.execute(sql)
    print(count)

    cur.scroll(0, mode='absolute')
    cur.close()
    conn.commit()
    conn.close()

if __name__ == '__main__':
    export('TestCase', r'./TestCase.xls')