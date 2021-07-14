import pandas as pd
from pandas import DataFrame
import datetime
import collections
import numpy as np
import numbers
import random
from pandas.tools.plotting import scatter_matrix
from itertools import combinations
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
import sys
import  importlib
importlib.reload(sys)
sys.path.append(path+"/Notes/07 申请评分卡中的数据预处理和特征衍生/")
from scorecard_fucntions import *
# -*- coding: utf-8 -*-

#####################################################
# Step 0: 读取与训练数据集具有相同结构的原始测试数据#
#####################################################

data1b = pd.read_csv('C://Users//glf//Desktop//two//第九讲数据//LogInfo_9w_2.csv', header = 0)
data2b = pd.read_csv('C://Users//glf//Desktop//two//第九讲数据//Kesci_Master_9w_gbk_2.csv', header = 0,encoding = 'gbk')
data3bb = pd.read_csv('C://Users//glf//Desktop//two//第九讲数据//Userupdate_Info_9w_2.csv', header = 0)

#############################################
# Step 1: 利用与训练数据集相同的方法导出特征#
#############################################

### 摘录每位申请人的申请日期
data1b['logInfo'] = data1b['LogInfo3'].map(lambda x: datetime.datetime.strptime(x,'%Y-%m-%d'))
data1b['Listinginfo'] = data1b['Listinginfo1'].map(lambda x: datetime.datetime.strptime(x,'%Y-%m-%d'))
data1b['ListingGap'] = data1b[['logInfo','Listinginfo']].apply(lambda x: (x[1]-x[0]).days,axis = 1)

'''
我们使用180作为最大时间窗来计算data1b中的一些特征。
使用时间窗口可设置为7天、30天、60天、90天、120天、150天、180天。
我们计算了在选定的时间窗内每个原始字段的总数和差异数。
'''
time_window = [7, 30, 60, 90, 120, 150, 180]
var_list = ['LogInfo1','LogInfo2']
data1bGroupbyIdx = pd.DataFrame({'Idx':data1b['Idx'].drop_duplicates()})

for tw in time_window:
    data1b['TruncatedLogInfo'] = data1b['Listinginfo'].map(lambda x: x + datetime.timedelta(-tw))
    temp = data1b.loc[data1b['logInfo'] >= data1b['TruncatedLogInfo']]
    for var in var_list:
        #统计LogInfo1和LogInfo2的频率
        count_stats = temp.groupby(['Idx'])[var].count().to_dict()
        data1bGroupbyIdx[str(var)+'_'+str(tw)+'_count'] = data1bGroupbyIdx['Idx'].map(lambda x: count_stats.get(x,0))

        # 计算LogInfo1和LogInfo2的不同值
        Idx_UserupdateInfo1 = temp[['Idx', var]].drop_duplicates()
        uniq_stats = Idx_UserupdateInfo1.groupby(['Idx'])[var].count().to_dict()
        data1bGroupbyIdx[str(var) + '_' + str(tw) + '_unique'] = data1bGroupbyIdx['Idx'].map(lambda x: uniq_stats.get(x,0))

        # 计算LogInfo1和LogInfo2中每个值的平均计数
        data1bGroupbyIdx[str(var) + '_' + str(tw) + '_avg_count'] = data1bGroupbyIdx[[str(var)+'_'+str(tw)+'_count',str(var) + '_' + str(tw) + '_unique']].\
            apply(lambda x: x[0]*1.0/x[1], axis=1)


data3b['ListingInfo'] = data3b['ListingInfo1'].map(lambda x: datetime.datetime.strptime(x,'%Y/%m/%d'))
data3b['UserupdateInfo'] = data3b['UserupdateInfo2'].map(lambda x: datetime.datetime.strptime(x,'%Y/%m/%d'))
data3b['ListingGap'] = data3b[['UserupdateInfo','ListingInfo']].apply(lambda x: (x[1]-x[0]).days,axis = 1)
data3b['UserupdateInfo1'] = data3b['UserupdateInfo1'].map(ChangeContent)
data3bGroupbyIdx = pd.DataFrame({'Idx':data3b['Idx'].drop_duplicates()})

time_window = [7, 30, 60, 90, 120, 150, 180]
for tw in time_window:
    data3b['TruncatedLogInfo'] = data3b['ListingInfo'].map(lambda x: x + datetime.timedelta(-tw))
    temp = data3b.loc[data3b['UserupdateInfo'] >= data3b['TruncatedLogInfo']]

    #更新的频率
    freq_stats = temp.groupby(['Idx'])['UserupdateInfo1'].count().to_dict()
    data3bGroupbyIdx['UserupdateInfo_'+str(tw)+'_freq'] = data3bGroupbyIdx['Idx'].map(lambda x: freq_stats.get(x,0))

    # 更新的类型数目
    Idx_UserupdateInfo1 = temp[['Idx','UserupdateInfo1']].drop_duplicates()
    uniq_stats = Idx_UserupdateInfo1.groupby(['Idx'])['UserupdateInfo1'].count().to_dict()
    data3bGroupbyIdx['UserupdateInfo_' + str(tw) + '_unique'] = data3bGroupbyIdx['Idx'].map(lambda x: uniq_stats.get(x, x))

    #每种类型的平均计数
    data3bGroupbyIdx['UserupdateInfo_' + str(tw) + '_avg_count'] = data3bGroupbyIdx[['UserupdateInfo_'+str(tw)+'_freq', 'UserupdateInfo_' + str(tw) + '_unique']]. \
        apply(lambda x: x[0] * 1.0 / x[1], axis=1)

    #申请人是否变更身份证号码、是否购车、婚姻状况、电话等事项
    Idx_UserupdateInfo1['UserupdateInfo1'] = Idx_UserupdateInfo1['UserupdateInfo1'].map(lambda x: [x])
    Idx_UserupdateInfo1_V2 = Idx_UserupdateInfo1.groupby(['Idx'])['UserupdateInfo1'].sum()
    for item in ['_IDNUMBER','_HASBUYCAR','_MARRIAGESTATUSID','_PHONE']:
        item_dict = Idx_UserupdateInfo1_V2.map(lambda x: int(item in x)).to_dict()
        data3bGroupbyIdx['UserupdateInfo_' + str(tw) + str(item)] = data3bGroupbyIdx['Idx'].map(lambda x: item_dict.get(x, x))

# 将上述特性与PPD_Training_Master_GBK_3_1_Training_Set中的原始特性结合起来
allData = pd.concat([data2b.set_index('Idx'), data3bGroupbyIdx.set_index('Idx'), data1bGroupbyIdx.set_index('Idx')],axis= 1)
# allData.to_csv(path+'/数据/bank default/allData_0_Test.csv',encoding = 'gbk')
allData.to_csv('C://Users//glf//Desktop//two//第九讲数据//allData_0_Test.csv',encoding = 'gbk')


#############################
# Step 2: 弥补缺失值连续变量#
#############################

#对字符串类型变量做一些更改，特别是将nan转换为nan，以便在映射字典中读取它
# testData = pd.read_csv(path+'/数据/bank default/allData_0_Test.csv',header = 0,encoding = 'gbk')
testData = pd.read_csv('C://Users//glf//Desktop//two//第九讲数据//allData_0_Test.csv',header = 0,encoding = 'gbk')
allData[col] = allData[col].map(lambda x: str(x).upper())
changedCols = ['WeblogInfo_20', 'UserInfo_17']
for col in changedCols:
    testData[col] = testData[col].map(lambda x: str(x).upper())

allFeatures = list(testData.columns)
allFeatures.remove('ListingInfo')
allFeatures.remove('target')
allFeatures.remove('Idx')


### 读取保存的WOE编码字典
fread = open(path+'/数据/bank default/var_WOE.pkl','r')
WOE_dict = pickle.load(fread)
fread.close()

### 下面的特性在步骤5中被选择到记分卡模型中
var_WOE_model = ['UserInfo_15_encoding_WOE', u'ThirdParty_Info_Period6_10_WOE', u'ThirdParty_Info_Period5_2_WOE', 'UserInfo_16_encoding_WOE', 'WeblogInfo_20_encoding_WOE',
            'UserInfo_7_encoding_WOE', u'UserInfo_17_WOE', u'ThirdParty_Info_Period3_10_WOE', u'ThirdParty_Info_Period1_10_WOE', 'WeblogInfo_2_encoding_WOE',
            'UserInfo_1_encoding_WOE']


#有些特性是分类类型的，我们需要对它们进行编码
var_encoding = [i.replace('_WOE','').replace('_encoding','') for i in var_WOE_model if i.find('_encoding')>=0]
for col in var_encoding:
    print(col)
    [col1, encode_dict] = encoded_features[col]
    testData[col1] = testData[col].map(lambda x: encode_dict.get(str(x),-99999))
    col2 = str(col1) + "_WOE"
    cutOffPoints = var_cutoff[col1]
    special_attribute = []
    if - 1 in cutOffPoints:
        special_attribute = [-1]
    binValue = testData[col1].map(lambda x: AssignBin(x, cutOffPoints, special_attribute=special_attribute))
    testData[col2] = binValue.map(lambda x: WOE_dict[col1][x])

#其他特性可以直接映射到WOE
var_others = [i.replace('_WOE','').replace('_encoding','') for i in var_WOE_model if i.find('_encoding') < 0]
for col in var_others:
    print(col)
    col2 = str(col) + "_WOE"
    if col in var_cutoff.keys():
        cutOffPoints = var_cutoff[col]
        special_attribute = []
        if - 1 in cutOffPoints:
            special_attribute = [-1]
        binValue = testData[col].map(lambda x: AssignBin(x, cutOffPoints, special_attribute=special_attribute))
        testData[col2] = binValue.map(lambda x: WOE_dict[col][x])
    else:
        testData[col2] = testData[col].map(lambda x: WOE_dict[col][x])


#制作设计矩阵
X = testData[var_WOE_model]
X['intercept'] = [1]*X.shape[0]
y = testData['target']


#加载训练模型
saveModel =open(path+'/数据/bank default/LR_Model_Normal.pkl','r')
LR = pickle.load(saveModel)
saveModel.close()

y_pred = LR.predict(X)

scorecard_result = pd.DataFrame({'prob':y_pred, 'target':y})
# 我们用KS和AR检验模型的性能
# 这两个指数都应该在30%以上
performance = KS_AR(scorecard_result,'prob','target')
print("KS and AR for the scorecard in the test dataset are %.0f%% and %.0f%%"%(performance['AR']*100,performance['KS']*100))
