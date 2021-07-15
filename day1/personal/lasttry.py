#!/usr/bin/env python
# -*- coding: utf-8 -*-
#@Time: 2021/4/13 10:10
#@Author: GLF
import pandas as pd
pd.set_option('display.max_columns',None)
pd.set_option('display.unicode.ambiguous_as_wide', True)
pd.set_option('display.unicode.east_asian_width', True)

data1 = pd.read_csv('C:\\Users\\glf\\Desktop\\测试医院\\住院字典表.txt',encoding='gbk')
data2 = pd.read_csv('C:\\Users\\glf\\Desktop\\测试医院\\门诊数据表.txt',names=['c0','c1','c2','c3','c4','c5','c6','c7','c8','c9','c10','c11','c12','c13','c14','c15',
                                                               'c16','c17','c18','c19','c20','c21','c22','c23','c24','c25','c26','c27','c28','c29','c30','c31',
                                                               'c32','c33','c34','c35','c36','c37','c38','c39','c40','c41','c42','c43','c44','c45',
                                                               'c46','c47','c48','c49','c50','c51','c52','c53','c54','c55','c56','c57','c58','c59'])


data1 = data1.values.tolist()
data2 = data2.values.tolist()
print(data1)
print(data2)
res = []
for i in data2:
        idcard = i[7]
        name = i[12]
        sbkh = i[44]
        print(idcard, name, sbkh)
        for j in data1:
            if i[12] == j[1] and i[44] == j[2]:
                i[7] = j[0]
                print(j)
                print(i)
                i[7] = str(i[7]) +'\t'
                res.append(i)
                break
        else:
            i[7] = str(i[7]) + '\t'
            res.append(i)
df = pd.DataFrame(res)
df.to_csv('C:\\Users\\glf\\Desktop\\99.csv',index=False,header=False)