#!/usr/bin/env python
# -*-coding:utf-8 -*-
# author:GLF time:2020.4.23

#索引从1开始排列
guige_df.index = guige_df.index+1

#过滤掉该字段下的内容包含*的,
df3['HIS名称'] = df3['HIS名称'].apply(lambda x:x.replace('*',""))

#查看项目分类字段下都有那些内容
print(guige_df['项目分类'].unique())

#根据某个字段进行去重
guige_df = guige_df.drop_duplicates(['收费单据号'],keep='last')

#调出某字段类型下住院的所有数据
guige_df = guige_df[guige_df['类型'] == '住院']

#可以给索引列加上序号
guige_df.columns.name = '序号'

#根据删除指定行,当没有inplace时,只在新的数据块会显示删除.
guige_df.drop(index = [19185,189077,372004,588431],inplace=True)

#非DATE类型时,调用此方法转为DATE类型
guige_df['处方日期'] = pd.to_datetime(guige_df['处方日期'])
guige_df = guige_df[guige_df['处方日期'] < '2019/10/01']

#读取数据时,skiprows代表跳过多少行,nrows代表读取多少行
guige_df = pd.read_csv('C:\\Users\\GUOLONGFEI\\Desktop\\人民医院门诊.csv',encoding='gbk',skiprows=800000,nrows=754500)

#按照|切分读取数据,计算字段相差时间,最后按照要求提取数据
guige_df = pd.read_csv('C:\\Users\\GUOLONGFEI\\Desktop\\xshyyy_1021.csv',encoding='gbk',sep='|')
guige_df['时间差']=(guige_df['出院日期'] - guige_df['入院日期']).dt.days
guige_df = guige_df[guige_df['时间差'] >= 1]

# 指定多列排序(注意：对Worthy列升序，再对Price列降序)，ascending不指定的话，默认是True升序
lists.sort_values(by=["Worthy","Price"],inplace=True,ascending=[True,False])

#合并两个文件里的内容,并自动去掉第二个文件的标题名
guige_df1 = pd.read_excel('C:\\Users\\GUOLONGFEI\\Desktop\\康复科筛查数据\\吞咽功能障碍训练.xlsx')
guige_df2 = pd.read_excel('C:\\Users\\GUOLONGFEI\\Desktop\\康复科筛查数据\\吞咽功能障碍训练1.xlsx')
guige_df3 = pd.concat([guige_df1,guige_df2],sort=False)

# 根据病种名称进行统计排序
dfs = guige_df["病种名称"].value_counts()
print(dfs)

#读取文件后对某列字段进行更改
guige_df = guige_df.rename(columns={'出库数':'出库数量','进销存药品名称':'999999'})

for groupname,grouplist in guige_df.groupby('药品及项目名称'):
    print(groupname)
    print(grouplist)

#根据某列不同内容输出到不同文件
df=pd.read_excel('C:\\Users\\glf\\Desktop\\HIS数据\\2020医院HIS数据.xlsx')
leixing_list = list(df['类型'].drop_duplicates())
for i in leixing_list:
    df1 = df[df['类型'] == i]
    df1.to_excel('C:\\Users\\glf\\Desktop\\2020\\%s.xlsx'%(i), index=False)

#对于常见的编码也无法进行解析的情况,可以试试这个
data_path='C:\\Users\\glf\\Desktop\\HIS2020_new.csv'
f = open(data_path, encoding='gbk',errors='ignore')
df = pd.read_csv(f,error_bad_lines=False,sep='\t')



















































