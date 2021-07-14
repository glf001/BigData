#!/usr/bin/env python
# -*- coding: utf-8 -*-
#@Time: 2021/6/23 18:15  
#@Author: GLF

# 步骤一：查看缺失值、重复值以及异常值，并清洗
# 步骤二：根据【id】对【name】列数据进行分组聚合
# 步骤三：设置最小支持度、最小置信度以及最小提升度
# 步骤四：调用apriori函数,并从算法返回结果中提取前件（head_set）、后件（tail_set)、支持度（support）、置信度（confidence）、提升度(lift)数据
# 步骤五：筛选满足“后件中占比较多”、“支持度较大”条件的数据

import matplotlib.pyplot as plt
from apyori import apriori
import pandas as pd
import warnings

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns',None)
pd.set_option('display.unicode.ambiguous_as_wide', True)
pd.set_option('display.unicode.east_asian_width', True)

# 定义函数,将数据类型转换成列表
df=pd.read_excel('C:\\Users\\glf\\Desktop\\test.xlsx')
def tolist(df):
    if isinstance(df,list):
        return df
    return [df]
# 定义函数，将数据类型转换成列表
df['name'] = df['name'].agg(tolist)
# 根据【id】对【name】列数据进行分组聚合并重置索引
new_df = df.groupby('id').sum().reset_index()
# print(new_df.head())

#设置最小支持度,最小置信度,最小提升度
min_support = 0.02
min_confidence = 0.45
min_lift = 1
#数据集(事务集合)
transactions = new_df['name']

#调用apriori函数,并从算法返回结果中提取前件,后件,支持度,置信度,提升度
results = apriori(transactions = transactions,min_support = min_support,min_confidence = min_confidence,min_lift= min_lift)
# for result in results:
#     print(result)

#创建提取结果列表
extract_result = []

for record in results:
    for orderedStatistic in record.ordered_statistics:
        # print(orderedStatistic)
        #前件
        items_base = orderedStatistic.items_base
        #后件
        items_add = orderedStatistic.items_add

        #跳过"前件"为空的数据
        if not items_base:
            continue
        #将frozenset 不可变集合转换成熟悉的字符串
        head_set = '{%s}' % ','.join(items_base)
        tail_set = '{%s}' % ','.join(items_add)
        #提取支持度,并保留3位小数
        support = round(record.support,3)
        #提取置信度,并保留3位小数
        confidence = round(orderedStatistic.confidence,3)
        #提取提升度,并保留3位小数
        lift = round(orderedStatistic.lift,3)
        #将提取的数据保存到提取列表中
        row = [head_set,tail_set,support,confidence,lift]
        extract_result.append(row)
#将数据转化为 DataFrame 的格式
result_df = pd.DataFrame(extract_result,columns=['前件','后件','支持度','置信度','提升度'])
# print(result_df.head())
# print(result_df.shape)
gel_pens = result_df[result_df['后件'] == '{中性笔}']
# print(gel_pens.head())
# print(gel_pens.shape)
gel_pens.sort_values('支持度',ignore_index=True,inplace=True)
print(gel_pens.head(9))

#设置柱宽
width = 0.2
#设置x/y坐标值
x = gel_pens.index

x1 = x - width / 2
y1 = gel_pens['支持度']

x2 = x + width / 2
y2 = gel_pens['置信度']

#设置中文字体
plt.rcParams['font.sans-serif']=['SimHei']    # 用来设置字体样式以正常显示中文标签
plt.rcParams['axes.unicode_minus']=False      # 默认是使用Unicode负号，设置正常显示字符，如正常显示负号
#设置画布尺寸
plt.figure(figsize = (16, 8))
#绘制多组柱状图
plt.bar(x1,y1,width = width,alpha = 0.8)
plt.bar(x2,y2,width = width,alpha = 0.8)
#设置图表标题名及字体大小
plt.title('"中性笔"对应前件的支持度,置信度数值比较',fontsize =30)
#设置坐标轴的标题名称及字体大小
plt.xlabel('前件',fontsize = 25)
plt.xlabel('数值',fontsize = 25)
#设置x坐标轴的刻度间隔,名称以及大小
plt.xticks(x,gel_pens['前件'], fontsize = 15)
#设置y坐标轴的刻度大小
plt.yticks(fontsize = 15)
#设置y坐标轴的数值显示范围
plt.ylim(bottom = 0,top = 0.6)
#设置图例
plt.legend(['支持度','置信度'],fontsize =15)
plt.show()









