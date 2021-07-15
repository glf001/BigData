#!/usr/bin/env python
# -*- coding: utf-8 -*-
#@Time: 2021/2/23 15:29  
#@Author: GLF

import pandas as pd

pd.set_option('display.max_columns',None)
pd.set_option('display.unicode.ambiguous_as_wide', True)
pd.set_option('display.unicode.east_asian_width', True)

data_path='C:\\Users\\glf\\Desktop\\医院HIS数据门诊2018年10月-12月.csv'
f = open(data_path, encoding='gbk',errors='ignore')
data = pd.read_csv(f)
#data = pd.read_csv('C:\\Users\\glf\\Desktop\\医院HIS数据门诊2018年10月-12月.csv',encoding='gbk')
print(data.head(5))
#df.to_csv('C:\\Users\\glf\\Desktop\\test.txt')