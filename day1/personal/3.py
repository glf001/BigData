#!/usr/bin/env python
# -*- coding: utf-8 -*-
#@Time: 2021/4/23 16:05  
#@Author: GLF

import pymysql as py
import pandas as pd

conn = py.connect(
            host='127.0.0.1',
            port=3306,
            user='root',
            passwd='root',
            db='xushui',
            charset='utf8')
cur = conn.cursor()
sql = "select * from dawu where HIS名称 like %s "
l_tupple = ['住院诊查费','布洛芬缓释胶囊','阿托伐他汀钙片']
# df = pd.read_sql(sql,con= conn)
# df.to_excel('C:\\Users\\glf\\Desktop\\测试.xlsx',index= False)
count=cur.executemany(sql,l_tupple)
print(count)
cur.close()
conn.commit()
conn.close()












