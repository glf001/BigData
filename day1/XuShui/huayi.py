#!/usr/bin/env python
# -*-coding:utf-8 -*-
# author:GLF time:2020.6.18

import pandas as pd
pd.set_option('display.max_columns',None)
pd.set_option('display.unicode.ambiguous_as_wide', True)
pd.set_option('display.unicode.east_asian_width', True)
guige_df = pd.read_csv('C:\\Users\\GUOLONGFEI\\Desktop\\xshyyy_1021.csv',encoding='gbk',sep='|')
# guige_df.insert(0, "时间差", None)
guige_df['处方日期'] = pd.to_datetime(guige_df['处方日期'])
guige_df['入院日期'] = pd.to_datetime(guige_df['入院日期'])
guige_df['出院日期'] = pd.to_datetime(guige_df['出院日期'])
guige_df = guige_df[guige_df['处方日期'] > '2019-01-01']
guige_df = guige_df[guige_df['处方日期'] < '2019-09-01']
guige_df = guige_df.drop_duplicates(['收费单据号'],keep='last')
# guige_df['时间差']=(guige_df['出院日期'] - guige_df['入院日期']).dt.days
# guige_df = guige_df[guige_df['时间差'] >= 1]
dfs = guige_df['单票费用合计'].sum()
dfss = guige_df['统筹支付'].sum()
print(dfs)
print(dfss)
print(guige_df.head())
print(guige_df.shape)