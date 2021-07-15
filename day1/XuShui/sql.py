#!/usr/bin/env python
# -*-coding:utf-8 -*-
# author:GLF time:2020.3.23

import os
import pymysql

def get_sql_files():
    sql_files = []
    files = os.listdir(os.path.dirname(os.path.abspath(__file__)))
    for file in files:
        if os.path.splitext(file)[1] == '.sql':
            sql_files.append(file)
    return sql_files

def connectMySQL():
    # 打开数据库连接
    db = pymysql.connect(
        host='127.0.0.1',
        port=3306,
        user='root',
        passwd='root',
        db='xushui',
        charset='utf8'
    )

    # 使用 cursor() 方法创建一个游标对象 cursor
    cursor = db.cursor()

    for file in get_sql_files():
        executeScriptsFromFile(file, cursor)
    db.close()


def executeScriptsFromFile(filename, cursor):
    fd = open(filename, 'r', encoding='utf-8')

    sqlFile = fd.read()
    fd.close()
    sqlCommands = sqlFile.split(';')

    for command in sqlCommands:
        try:
            cursor.execute(command)
        except Exception as msg:
            print(msg)

    print('sql执行完成')


if __name__ == "__main__":
    connectMySQL()




# import pymysql.cursors
# import xlwt
# import sys
# import importlib
# importlib.reload(sys)
# def export(table_name,outputpath):
#     conn = pymysql.connect(
#         host='192.168.1.159',
#         port=3306,
#         user='root',
#         passwd='root',
#         db='xushui',
#         charset='utf8')
#     cur = conn.cursor()
#     sql = "select * from rmyy where his名称 like '%一次性注射器%'"
#
#     count = cur.execute(sql)
#     print(count)
#     cur.scroll(0, mode='absolute')
#     result = cur.fetchall()
#     fields = cur.description
#     workbook = xlwt.Workbook()
#     sheet = workbook.add_sheet(table_name, cell_overwrite_ok=True)
#
#     # 写上字段信息
#     for field in range(0, len(fields)):
#         sheet.write(0, field, fields[field][0])
#
#     # 获取并写入数据段信息
#     row = 1
#     col = 0
#     for row in range(1, len(result) + 1):
#         for col in range(0, len(fields)):
#             sheet.write(row, col, u'%s' % result[row - 1][col])
#
#     workbook.save(outputpath)
#
#     cur.close()
#     conn.commit()
#     conn.close()
#
# if __name__ == '__main__':
#     export('TestCase', r'./TestCase.xls')


# sql = "select t4.* from huayi3 t4, " \
#           "(select DISTINCT t1.处方日期,t1.个人编号 from " \
#           "(select 处方日期,个人编号 from huayi3 where his名称 = '血清碳酸氢盐(HCO3)测定酶促动力学法') t1," \
#           "(select 处方日期,个人编号 from huayi3 where his名称 = '无机磷测定电极法') t2," \
#           "(select 处方日期,个人编号 from huayi3 where his名称 = '钙测定离子选择电极法') t3," \
#           "(select 处方日期,个人编号 from huayi3 where his名称 = '钾测定火焰分光光度法或离子选择电极法') t6," \
#           "(select 处方日期,个人编号 from huayi3 where his名称 = '氯测定离子选择电极法') t7," \
#           "(select 处方日期,个人编号 from huayi3 where his名称 = '钠测定火焰分光光度法或离子选择电极法') t8," \
#           "(select 处方日期,个人编号 from huayi3 where his名称 = '镁测定分光光度法') t9" \
#           " where t1.处方日期 = t2.处方日期 and t1.处方日期 = t3.处方日期 and " \
#           "       t1.处方日期 = t6.处方日期 and t1.处方日期 = t7.处方日期 and " \
#           "       t1.处方日期 = t8.处方日期 and t1.处方日期 = t9.处方日期 and " \
#           "       t1.个人编号 = t2.个人编号 and t1.个人编号 = t3.个人编号 and " \
#           "       t1.个人编号 = t6.个人编号 and t1.个人编号 = t7.个人编号 and " \
#           "       t1.个人编号 = t8.个人编号 and t1.个人编号 = t9.个人编号 ) t5 " \
#           " where t5.处方日期 = t4.处方日期 and t5.个人编号 = t4.个人编号 and " \
#           "(t4.his名称 = '血清碳酸氢盐(HCO3)测定酶促动力学法'" \
#           "or t4.his名称 = '无机磷测定电极法'" \
#           "or t4.his名称 = '钙测定离子选择电极法'" \
#           "or t4.his名称 = '钾测定火焰分光光度法或离子选择电极法'" \
#           "or t4.his名称 = '氯测定离子选择电极法'" \
#           "or t4.his名称 = '钠测定火焰分光光度法或离子选择电极法'" \
#           "or t4.his名称 = '镁测定分光光度法')"