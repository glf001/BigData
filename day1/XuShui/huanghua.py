#!/usr/bin/env python
# -*-coding:utf-8 -*-
# author:GLF time:2020.7.9

import pandas as pd
import pandas_profiling
pd.set_option('display.max_columns',None)
pd.set_option('display.unicode.ambiguous_as_wide', True)
pd.set_option('display.unicode.east_asian_width', True)

#guige_df = pd.read_csv('C:\\Users\\GUOLONGFEI\\Desktop\\2017.csv',encoding='utf-8',sep='\t')
guige_df = pd.read_excel('C:\\Users\\GUOLONGFEI\\Desktop\\住院2019-1.xlsx',encoding='utf-8')

print(guige_df.head())
print(guige_df.shape)

if __name__ == '__main__':
   pfr = pandas_profiling.ProfileReport(guige_df)
   pfr.to_file("./test.html")

