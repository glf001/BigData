#!/usr/bin/env python
# -*-coding:utf-8 -*-
# author:GLF time:2019.8.23

import pandas as pd

pd.set_option('display.max_columns',None)
pd.set_option('display.unicode.ambiguous_as_wide', True)
pd.set_option('display.unicode.east_asian_width', True)
guige_df = pd.read_excel('C:\\Users\\GUOLONGFEI\\Desktop\\品规名1.xlsx')

# 统计出采购金额排名前10的品种
def money(guige_df):
    guige_df = guige_df.sort_values(["采购金额（元）"], ascending=False)
    guige_df = guige_df.head(10)
    print(guige_df.shape)
    print([column for column in guige_df])
    guige_df = pd.DataFrame(guige_df)
    print("=============================以下是采购金额top10排名")
    print(guige_df.loc[:, ['品种名', '采购金额（元）']])
money(guige_df)

#统计出采购量排名前10的品种
def count(guige_df):
    guige_df = guige_df.sort_values(["采购量"], ascending=False)
    guige_df = guige_df.head(10)
    guige_df = pd.DataFrame(guige_df)
    print("==============================以下是采购量top10排名")
    print(guige_df.loc[:, ['品种名', '采购量']])
count(guige_df)

#按照采购金额降序排序，取前1%的为大品种药
def breed(guige_df):
    guige_df = guige_df.sort_values(["采购金额（元）"], ascending=False)
    print(guige_df.shape)
    print([column for column in guige_df])
    guige_df = pd.DataFrame(guige_df)
    print("=============================以下是采购金额占前10%的大品种药")
    print(guige_df.loc[:, ['品种名', '采购金额（元）', '金额占比（%）']])
breed(guige_df)

