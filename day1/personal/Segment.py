#!/usr/bin/env python
# -*-coding:utf-8 -*-
# author:GLF time:2020.8.23

import pandas as pd

pd.set_option('display.max_columns',None)
pd.set_option('display.unicode.ambiguous_as_wide', True)
pd.set_option('display.unicode.east_asian_width', True)

data_path='C:\\Users\\glf\\Desktop\\2020.csv'
f = open(data_path, encoding='utf',errors='ignore')
df = pd.read_csv(f,error_bad_lines=False)
del df['序号']
grade=list(set(df['医院名称']))
#不保存列名header=0
for g in grade:
    df[df['医院名称']==g].to_csv('C:\\Users\\glf\\Desktop\\2020\\'+str(g)+'.csv',index=False)