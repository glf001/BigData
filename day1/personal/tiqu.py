#!/usr/bin/env python
# -*- coding: utf-8 -*-
#@Time: 2021/1/5 17:11  
#@Author: GLF

import pandas as pd
import datetime

pd.set_option('display.max_columns',None)
pd.set_option('display.unicode.ambiguous_as_wide', True)
pd.set_option('display.unicode.east_asian_width', True)
# print(datetime.datetime.now())
data_path='C:\\Users\\glf\\Desktop\\门诊2020.csv'
f = open(data_path, encoding='gbk',errors='ignore')
df = pd.read_csv(f,error_bad_lines=False,sep='\t')
print(df.head())
print(df.shape)
df.to_csv('C:\\Users\\glf\\Desktop\\门诊2020zyy_his.csv',index=False)

# df=pd.read_csv('C:\\Users\\glf\\Desktop\\中创住院.csv',encoding='gb18030')
# df['检验日期'] = pd.to_datetime(df['检验日期'])
# df['检验日期'] = df['检验日期'].dt.strftime('%Y-%m-%d %H:%M:%S')
#df = df.rename(columns=lambda x: x.replace("\n","").replace(",",";")).replace(" ","")
# df.replace('\n+','',regex=True,inplace=True)
# df.replace(',',';',regex=True,inplace=True)
# df.replace('，',';',regex=True,inplace=True)
# print(datetime.datetime.now())
# df1 = df[df['医院名称'] == '东光县医院']
# df2 = df[df['医院名称'] == '东光县中医院']
# df1.to_csv('C:\\Users\\glf\\Desktop\\东光县医院2020医保数据.csv',index=False)
# df2.to_csv('C:\\Users\\glf\\Desktop\\东光县中医院2020医保数据.csv',index=False)