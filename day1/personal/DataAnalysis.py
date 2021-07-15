#!/usr/bin/env python
# -*-coding:utf-8 -*-
# author:GLF time:2020.6.1

import pandas as pd
import difflib as diff

pd.set_option('display.max_columns',None)
pd.set_option('display.unicode.ambiguous_as_wide', True)
pd.set_option('display.unicode.east_asian_width', True)

def similarity():
    df = pd.read_excel('C:\\Users\\glf\\Desktop\\东光县秦村镇中心卫生院医疗目录对照表.xlsx')
    df['相似度'] = df.apply(lambda x: diff.SequenceMatcher(None,x[1].strip(),x[4].strip()).ratio(),axis=1)
    # df = df[df['相似度'] < 0.8]
    print(df.head())
    print(df.shape)
    df.to_excel('C:\\Users\\glf\\Desktop\\东光县秦村镇中心卫生院医疗目录对照表新.xlsx',index=False)

def comparison_table():
    df1 = pd.read_csv('C:\\Users\\GUOLONGFEI\\Desktop\\职工数据.csv', encoding='gb18030')
    df2 = pd.read_excel('C:\\Users\\GUOLONGFEI\\Desktop\\耗材超价.xlsx')
    df3 = pd.merge(df1, df2, how='inner', on=["HIS名称"], right_index=True)
    df3['HIS名称'] = df3['HIS名称'].apply(lambda x: x.replace('*', ""))
    df3['HIS名称'] = df3['HIS名称'].apply(lambda x: x.replace('<', ""))
    df3['HIS名称'] = df3['HIS名称'].apply(lambda x: x.replace('>', ""))
    df3['HIS名称'] = df3['HIS名称'].apply(lambda x: x.replace('?', ""))
    df3.insert(0, "扣费", None)
    df3['扣费'] = (df3['单价'] - df3['符合规定的加成后售价'])*df3['数量']
    print(df3.head())
    print(df3.shape)
    his_name = list(set(df3['HIS名称']))
    # df3.to_csv(r'C:/Users/GUOLONGFEI/Desktop/对照.csv', encoding='gbk', index=False)
    for his in his_name:
        df3[df3['HIS名称'] == his].to_csv(r'C:/Users/GUOLONGFEI/Desktop/耗材筛查/' + his + '1.csv', encoding='gb18030',index=False)

def Hospitals_are_linked_to_health_insurance():
    df1 = pd.read_excel('C:\\Users\\GUOLONGFEI\\Desktop\\医保氨曲南.xlsx')
    df2 = pd.read_csv('C:\\Users\\GUOLONGFEI\\Desktop\\lis.csv')
    # df3 = pd.merge(df1, df2, how='inner', on=["姓名","HIS名称","住院号"], right_index=True)
    df3 = pd.merge(df1, df2, how='inner', on=["姓名"], right_index=True)
    print(df3.head())
    print(df3.shape)
    df3.to_excel(r'C:/Users/GUOLONGFEI/Desktop/润泽门诊对照.xlsx',index=False)

#求相似度算法
if __name__ == '__main__':
    similarity()

#通过医院与医保对照表对照,在关联匹配出医保数据中的相关数据
# if __name__ == '__main__':
#     comparison_table()

#查出医保数据问题后,需要关联医院his进行辅助查询
# if __name__ == '__main__':
#     Hospitals_are_linked_to_health_insurance()
