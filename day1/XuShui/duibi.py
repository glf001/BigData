#!/usr/bin/env python
# -*-coding:utf-8 -*-
# author:GLF time:2019.11.21
def function(a, b):
    if a == b:
        return 1
    else:
        return 0
import pandas as pd
pd.set_option('display.max_columns',None)
pd.set_option('display.unicode.ambiguous_as_wide', True)
pd.set_option('display.unicode.east_asian_width', True)
guige_df = pd.read_csv('C:\\Users\\GUOLONGFEI\\Desktop\\徐水中医院职工诊疗.csv',encoding='gb18030')
print(guige_df.head())
df = pd.DataFrame(guige_df)
df['bool'] = df.apply(lambda x : function(x[u'项目名称'],x[u'医院内名称']),axis = 1)
print(df.head(10))
df.to_excel('C:\\Users\\GUOLONGFEI\\Desktop\\处理后徐水中医院职工诊疗.xls')