#!/usr/bin/env python
# -*-coding:utf-8 -*-
# author:GLF time:2019.9.3

import pandas as pd
import numpy as np

pd.set_option('mode.chained_assignment', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.unicode.ambiguous_as_wide', True)
pd.set_option('display.unicode.east_asian_width', True)
guige_df = pd.read_excel('C:\\Users\\GUOLONGFEI\\Desktop\\品规名1.xlsx')

def Marker(guige_df):
    guige_df.insert(0, "市场集中度", None)
    df = pd.DataFrame()
    for name, group in guige_df.groupby("年月"):
        print(name)
        n = group["年月"].count()
        x = group["采购金额（元）"].aggregate(np.sum)
        print("=====================================================================")
        for name, group in group.groupby("企业名称"):
            print(name)
            print(group["采购金额（元）"])
            si = (group["采购金额（元）"] / x) ** 2
            si2 = si * 10000
            group["市场集中度"] = si2
            df = df.append(group)
        df.to_excel('C:\\Users\\GUOLONGFEI\\Desktop\\市场集中度.xlsx')
Marker(guige_df)