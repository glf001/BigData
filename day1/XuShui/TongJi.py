import pymysql.cursors
import xlwt
import sys
import importlib
import pandas as pd
import numpy as np
importlib.reload(sys)

def export(table_name,outputpath):
    pd.set_option('display.max_columns', None)
    pd.set_option('display.unicode.ambiguous_as_wide', True)
    pd.set_option('display.unicode.east_asian_width', True)
    conn = pymysql.connect(
        host='127.0.0.1',
        port=3306,
        user='root',
        passwd='root',
        db='handan',
        charset='utf8')
    cur = conn.cursor()
    sql = "select * from mryy2 where 科室名称 like '%康复%' and his名称 = '运动疗法'"

    df = pd.read_sql(sql,con= conn)
    df.insert(0, "数量合", None)
    df.insert(0, "金额合", None)
    df.insert(0, "条数", None)
    df["数量"] = df["数量"].astype('float')
    df["金额"] = df["金额"].astype('float')
    df["数量合"] = df["数量"].sum()
    df["金额合"] = df["金额"].sum()
    df["条数"] = df["HIS名称"].count()
    df.to_csv('C:\\Users\\GUOLONGFEI\\Desktop\\康复科筛查数据\\运动疗法.csv',encoding='gbk',index=False,mode='a',header=False)
    count = cur.execute(sql)
    print(count)
    cur.scroll(0, mode='absolute')
    result = cur.fetchall()
    # fields = cur.description
    # workbook = xlwt.Workbook()
    # sheet = workbook.add_sheet(table_name, cell_overwrite_ok=True)

    # 写上字段信息
    # for field in range(0, len(fields)):
    #     sheet.write(0, field, fields[field][0])
    #
    # # 获取并写入数据段信息
    # row = 1
    # col = 0
    # for row in range(1, len(result) + 1):
    #     for col in range(0, len(fields)):
    #         sheet.write(row, col, u'%s' % result[row - 1][col])
    #
    # workbook.save(outputpath)
    cur.close()
    conn.commit()
    conn.close()

if __name__ == '__main__':
    export('TestCase', r'./TestCase.xls')