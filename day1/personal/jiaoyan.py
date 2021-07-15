#!/usr/bin/env python
# -*- coding: utf-8 -*-
#@Time: 2021/4/21 9:13  
#@Author: GLF

import pandas as pd
pd.set_option('display.max_columns',None)
pd.set_option('display.unicode.ambiguous_as_wide', True)
pd.set_option('display.unicode.east_asian_width', True)

# data_path='C:\\Users\\glf\\Desktop\\PACS12.csv'
# f = open(data_path, encoding='gb18030',errors='ignore')
# df = pd.read_csv(f,error_bad_lines=False,names=['类型','医院定点编码','医院名称','医院级别','病种代码','病种名称','科室代码','科室名称','患者身份证号','患者姓名','患者性别','患者年龄','患者出生日期','人员类别','社保卡号','收费单据号','项目唯一编码',
#                                                                '药品及项目名称','剂型','规格','单价','数量','中药饮片付数','金额','单项医保统筹支付','医保限价支付标准','限制用药情况','单票费用合计','自费','自付','基本账户','单票医保统筹支付','处方日期',
#                                                                '住院号','入院诊断','出院诊断','入院日期','出院日期','中心编码','中心名称','HIS内码','HIS名称','个人编号','医保支付甲乙丙类','单位','药品诊疗床位费','项目分类',
#                                                                '医生编码','医生名称','就医人员分类','登记流水号','是否退票','药品或医用耗材生产厂家','医嘱内容','是否耗材','错误1','错误2','错误3','错误4'])
# del df['序号']
df = pd.read_csv('E:\\东光地区\\dgyp2018_1\\HIS数据2018_0_right_5.csv')
df = pd.read_csv(f,error_bad_lines=False,sep='|')
df.index = df.index + 1
df.insert(0, "序号", None)
df['序号'] = df.index + 1
print(df.head())
print(df.shape)
# df.to_csv('C:\\Users\\glf\\Desktop\\最新HIS2018.csv',encoding='utf',sep='\t')

# data = pd.read_csv('C:\\Users\\glf\\Desktop\\东光县医院按医院his名称进行关联结果.csv')
# row_num, column_num = data.shape  # 数据共有多少行，多少列
# print('the sample number is %s and the column number is %s' % (row_num, column_num))
# # 这里我们的数据共有210000行，假设要让每个文件10万行数据
# for i in range(0, 3):
#     #save_data = data.iloc[i * 100000 + 1:(i + 1) * 100000 + 1, :]  # 每隔10万循环一次
#     save_data = data.iloc[i * 500000:(i + 1) * 500000, :]  # 每隔20万循环一次
#     file_name = 'C:\\Users\\glf\\Desktop\\博士按政策筛查的关联' + str(i) + '.xlsx'
#     save_data.to_excel(file_name,index=False)
