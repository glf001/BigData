#!/usr/bin/env python
# -*- coding: utf-8 -*-
#@Time: 2021/1/12 14:24  
#@Author: GLF

import pandas as pd
import matplotlib.pyplot as plt

pd.set_option('display.max_columns',None)
pd.set_option('display.unicode.ambiguous_as_wide', True)
pd.set_option('display.unicode.east_asian_width', True)

df=pd.read_csv('C:\\Users\\glf\\Desktop\\dawu.csv')
fig=plt.figure() #Plots in matplotlib reside within a figure object, use plt.figure to create new figure
#Create one or more subplots using add_subplot, because you can't create blank figure
ax = fig.add_subplot(1,1,1)
#Variable
ax.hist(df['年龄'],bins = 10) # Here you can play with number of bins
# Labels and Tit
plt.title('Age distribution')
plt.xlabel('Age')
plt.ylabel('#nums')
plt.show()
