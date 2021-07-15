#!/usr/bin/env python
# -*-coding:utf-8 -*-
# author:GLF time:2019.9.9

import pandas as pd
import numpy as np

pd.set_option('mode.chained_assignment', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.unicode.ambiguous_as_wide', True)
pd.set_option('display.unicode.east_asian_width', True)
guige_df = pd.read_excel('C:\\Users\\GUOLONGFEI\\Desktop\\品规名1.xlsx')

def analyse(guige_df):
    # guige_df.insert(0,"占比1",None)
    # guige_df["占比1"] = guige_df.get("采购金额（元）") / 12345
    # print(np.std(guige_df["采购量"].values))
    # print(np.mean(guige_df["采购量"].values))
    guige_df.insert(0,"采购量对数值",None)
    guige_df["采购量对数值"] = np.log(guige_df["采购量"])
    guige_df.insert(0, "不稳定系数", None)
    df = pd.DataFrame()
    for name, group in guige_df.groupby("品种名"):
        print(name)
        cv = np.std(group["采购量"].values) / np.mean(group["采购量"].values)
        num = 12
        for i in group["采购量"].values:
            if i == 0:
                num -= 1
        d = 1 / (num / 12)
        group["不稳定系数"] = np.log(cv * d)
        df = df.append(group)
    df.to_excel('C:\\Users\\GUOLONGFEI\\Desktop\\不稳定分析.xlsx')
analyse(guige_df)