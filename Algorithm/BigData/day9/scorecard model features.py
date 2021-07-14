import pandas as pd
import datetime
import collections
import numpy as np
import numbers
import random
from itertools import combinations
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
import sys
import pickle
import importlib
importlib.reload(sys)
# sys.path.append(path+"/Notes/07 申请评分卡中的数据预处理和特征衍生/")
from scorecard_fucntions import *
from sklearn.linear_model import LogisticRegressionCV
# -*- coding: utf-8 -*-

############################################################
#Step 0: 启动数据处理工作，包括读取csv文件，检查Idx的一致性#
############################################################

data1 = pd.read_csv('C://Users//glf//Desktop//two//第六讲数据//PPD_LogInfo_3_1_Training_Set.csv', header = 0)
data2 = pd.read_csv('C://Users//glf//Desktop//two//第六讲数据//PPD_Training_Master_GBK_3_1_Training_Set.csv', header = 0,encoding = 'gbk')
data3 = pd.read_csv('C://Users//glf//Desktop//two//第六讲数据//PPD_Userupdate_Info_3_1_Training_Set.csv', header = 0)

data1_Idx, data2_Idx, data3_Idx = set(data1.Idx), set(data2.Idx), set(data3.Idx)
check_Idx_integrity = (data1_Idx - data2_Idx)|(data2_Idx - data1_Idx)|(data1_Idx - data3_Idx)|(data3_Idx - data1_Idx)
print(check_Idx_integrity)

#set([85832, 82505, 10922, 78259, 14662]) 只在data1_Idx中，所以我们将它们从建模基中删除

##################################################################################################################################
# Step 1:使用PPD_Training_Master_GBK_3_1_Training_Set、PPD_LogInfo_3_1_Training_Set和PPD_Userupdate_Info_3_1_Training_Set派生特征#
##################################################################################################################################
# 比较三个city变量是否匹配
data2['city_match'] = data2.apply(lambda x: int(x.UserInfo_2 == x.UserInfo_4 == x.UserInfo_8 == x.UserInfo_20),axis = 1)
del data2['UserInfo_2']
del data2['UserInfo_4']
del data2['UserInfo_8']
del data2['UserInfo_20']

#摘录每位申请人的申请日期
#登陆时间
data1['logInfo'] = data1['LogInfo3'].map(lambda x: datetime.datetime.strptime(x,'%Y-%m-%d'))
#借款成交时间
data1['Listinginfo'] = data1['Listinginfo1'].map(lambda x: datetime.datetime.strptime(x,'%Y-%m-%d'))
#借款成交时间-登陆时间
data1['ListingGap'] = data1[['logInfo','Listinginfo']].apply(lambda x: (x[1]-x[0]).days,axis = 1)

#maxListingGap = max(data1['ListingGap'])
#查看不同时间切片的覆盖度，发现180天时，覆盖度达到95%
timeWindows = TimeWindowSelection(data1, 'ListingGap', range(30,361,30))

'''
我们使用180作为最大时间窗来计算data1中的一些特征。
使用时间窗口可设置为7天、30天、60天、90天、120天、150天、180天。
我们计算了在选定的时间窗内每个原始字段的总数和差异数。
'''
time_window = [7, 30, 60, 90, 120, 150, 180]
#可以衍生特征
var_list = ['LogInfo1','LogInfo2']
data1GroupbyIdx = pd.DataFrame({'Idx':data1['Idx'].drop_duplicates()})

for tw in time_window:
    data1['TruncatedLogInfo'] = data1['Listinginfo'].map(lambda x: x + datetime.timedelta(-tw))
    temp = data1.loc[data1['logInfo'] >= data1['TruncatedLogInfo']]
    for var in var_list:
        #统计LogInfo1和LogInfo2的频率
        count_stats = temp.groupby(['Idx'])[var].count().to_dict()
        data1GroupbyIdx[str(var)+'_'+str(tw)+'_count'] = data1GroupbyIdx['Idx'].map(lambda x: count_stats.get(x,0))

        # 计算LogInfo1和LogInfo2的不同值
        Idx_UserupdateInfo1 = temp[['Idx', var]].drop_duplicates()
        uniq_stats = Idx_UserupdateInfo1.groupby(['Idx'])[var].count().to_dict()
        data1GroupbyIdx[str(var) + '_' + str(tw) + '_unique'] = data1GroupbyIdx['Idx'].map(lambda x: uniq_stats.get(x,0))

        # 计算LogInfo1和LogInfo2中每个值的平均计数
        data1GroupbyIdx[str(var) + '_' + str(tw) + '_avg_count'] = data1GroupbyIdx[[str(var)+'_'+str(tw)+'_count',str(var) + '_' + str(tw) + '_unique']].\
            apply(lambda x: x[0]*1.0/x[1], axis=1)


data3['ListingInfo'] = data3['ListingInfo1'].map(lambda x: datetime.datetime.strptime(x,'%Y/%m/%d'))
data3['UserupdateInfo'] = data3['UserupdateInfo2'].map(lambda x: datetime.datetime.strptime(x,'%Y/%m/%d'))
data3['ListingGap'] = data3[['UserupdateInfo','ListingInfo']].apply(lambda x: (x[1]-x[0]).days,axis = 1)
collections.Counter(data3['ListingGap'])
hist_ListingGap = np.histogram(data3['ListingGap'])
hist_ListingGap = pd.DataFrame({'Freq':hist_ListingGap[0],'gap':hist_ListingGap[1][1:]})
hist_ListingGap['CumFreq'] = hist_ListingGap['Freq'].cumsum()
hist_ListingGap['CumPercent'] = hist_ListingGap['CumFreq'].map(lambda x: x*1.0/hist_ListingGap.iloc[-1]['CumFreq'])

'''
我们使用180作为最大时间窗来计算data1中的一些特征。所使用的时间窗口可以设置为
7天，30天，60天，90天，120天，150天，180天
因为我们观察到一些字母的大小写不匹配，比如QQ和QQ, Idnumber和Idnumber，所以我们首先让它们保持一致。
此外，我们把手机和电话组合成电话。
在选定的时间窗口内，我们计算
(1)更新频率
(2)每个项目的区别
(3)一些重要的物品，如身份证号码，购车号码，婚姻登记号码，电话
'''
data3['UserupdateInfo1'] = data3['UserupdateInfo1'].map(ChangeContent)
data3GroupbyIdx = pd.DataFrame({'Idx':data3['Idx'].drop_duplicates()})

time_window = [7, 30, 60, 90, 120, 150, 180]
for tw in time_window:
    data3['TruncatedLogInfo'] = data3['ListingInfo'].map(lambda x: x + datetime.timedelta(-tw))
    temp = data3.loc[data3['UserupdateInfo'] >= data3['TruncatedLogInfo']]

    #更新的频率
    freq_stats = temp.groupby(['Idx'])['UserupdateInfo1'].count().to_dict()
    data3GroupbyIdx['UserupdateInfo_'+str(tw)+'_freq'] = data3GroupbyIdx['Idx'].map(lambda x: freq_stats.get(x,0))

    # 更新的类型数目
    Idx_UserupdateInfo1 = temp[['Idx','UserupdateInfo1']].drop_duplicates()
    uniq_stats = Idx_UserupdateInfo1.groupby(['Idx'])['UserupdateInfo1'].count().to_dict()
    data3GroupbyIdx['UserupdateInfo_' + str(tw) + '_unique'] = data3GroupbyIdx['Idx'].map(lambda x: uniq_stats.get(x, x))

    #每种类型的平均计数
    data3GroupbyIdx['UserupdateInfo_' + str(tw) + '_avg_count'] = data3GroupbyIdx[['UserupdateInfo_'+str(tw)+'_freq', 'UserupdateInfo_' + str(tw) + '_unique']]. \
        apply(lambda x: x[0] * 1.0 / x[1], axis=1)

    #申请人是否变更身份证号码、是否购车、婚姻状况、电话等事项
    Idx_UserupdateInfo1['UserupdateInfo1'] = Idx_UserupdateInfo1['UserupdateInfo1'].map(lambda x: [x])
    Idx_UserupdateInfo1_V2 = Idx_UserupdateInfo1.groupby(['Idx'])['UserupdateInfo1'].sum()
    for item in ['_IDNUMBER','_HASBUYCAR','_MARRIAGESTATUSID','_PHONE']:
        item_dict = Idx_UserupdateInfo1_V2.map(lambda x: int(item in x)).to_dict()
        data3GroupbyIdx['UserupdateInfo_' + str(tw) + str(item)] = data3GroupbyIdx['Idx'].map(lambda x: item_dict.get(x, x))

# 将上述特性与PPD_Training_Master_GBK_3_1_Training_Set中的原始特性结合起来
allData = pd.concat([data2.set_index('Idx'), data3GroupbyIdx.set_index('Idx'), data1GroupbyIdx.set_index('Idx')],axis= 1)
allData.to_csv('C://Users//glf//Desktop//two//第六讲数据//allData_0.csv',encoding = 'gbk')

#########################################
# Step 2: 弥补分类变量和连续变量的缺失值#
#########################################
allData = pd.read_csv('C://Users//glf//Desktop//two//第六讲数据//allData_0.csv',header = 0,encoding = 'gbk')
allFeatures = list(allData.columns)
allFeatures.remove('ListingInfo')
allFeatures.remove('target')
allFeatures.remove('Idx')

#检查列并删除它们，如果它们是常量
for col in allFeatures:
    if len(set(allData[col])) == 1:
        del allData[col]
        allFeatures.remove(col)

#将整个自变量分为范畴型和数值型
numerical_var = []
for var in allFeatures:
    uniq_vals = list(set(allData[var]))
    if np.nan in uniq_vals:
        uniq_vals.remove( np.nan)
    if len(uniq_vals) >= 10 and isinstance(uniq_vals[0],numbers.Real):
        numerical_var.append(var)

categorical_var = [i for i in allFeatures if i not in numerical_var]

'''
对于每个类别变量，如果缺失的值占据50%以上，我们将其删除。
否则我们将使用missing作为特殊状态
'''
missing_pcnt_threshould_1 = 0.5
for col in categorical_var:
    missingRate = MissingCategorial(allData,col)
    print('{0} has missing rate as {1}'.format(col,missingRate))
    if missingRate > missing_pcnt_threshould_1:
        categorical_var.remove(col)
        del allData[col]
    if 0 < missingRate < missing_pcnt_threshould_1:
        allData[col] = allData[col].map(lambda x: str(x).upper())


'''
对于连续变量，如果缺失值大于30%，我们将其移除。
否则，我们采用随机抽样的方法来弥补缺失
'''
missing_pcnt_threshould_2 = 0.3
for col in numerical_var:
    missingRate = MissingContinuous(allData, col)
    print('{0} has missing rate as {1}'.format(col, missingRate))
    if missingRate > missing_pcnt_threshould_2:
        numerical_var.remove(col)
        del allData[col]
        print('we delete variable {} because of its high missing rate'.format(col))
    else:
        if missingRate > 0:
            not_missing = allData.loc[allData[col] == allData[col]][col]
            makeuped = allData[col].map(lambda x: MakeupRandom(x, list(not_missing)))
            del allData[col]
            allData[col] = makeuped
            missingRate2 = MissingContinuous(allData, col)
            print('missing rate after making up is:{}'.format(str(missingRate2)))

allData.to_csv('C://Users//glf//Desktop//two//第六讲数据//allData_1b.csv', header=True,encoding='gbk', columns = allData.columns, index=False)

#############################
# Step 3: 将变量分组到容器中#
#############################
#对于每个类别变量，如果它有不同的值超过5，我们使用ChiMerge来合并它

trainData = pd.read_csv('C://Users//glf//Desktop//two//第六讲数据//allData_1b.csv',header = 0, encoding='gbk')
allFeatures = list(trainData.columns)
allFeatures.remove('ListingInfo')
allFeatures.remove('target')
allFeatures.remove('Idx')
#将整个自变量分为范畴型和数值型
numerical_var = []
for var in allFeatures:
    uniq_vals = list(set(trainData[var]))
    if np.nan in uniq_vals:
        uniq_vals.remove( np.nan)
    if len(uniq_vals) >= 10 and isinstance(uniq_vals[0],numbers.Real):
        numerical_var.append(var)

categorical_var = [i for i in allFeatures if i not in numerical_var]

for col in categorical_var:
    trainData[col] = trainData[col].map(lambda x: str(x).upper())


'''
对于谨慎变量，请遵循以下步骤
1，如果变量的不同值大于5，我们计算坏率，并用坏率对变量进行编码
2,否则:
(2.1)检查最大bin，如果最大bin占用率超过90%，则删除该变量
(2.2)检查每一箱的坏样率，如果任意一箱有0个坏样，将其与最小的非零坏样结合，
然后再检查最大的箱子
'''
deleted_features = []  #删除其中一个分类特征占比超过90%
encoded_features = {}
merged_features = {}
var_IV = {}  #保存被装箱特性的IV值
var_WOE = {}
for col in categorical_var:
    print('we are processing {}'.format(col))
    if len(set(trainData[col]))>5:
        print('{} is encoded with bad rate'.format(col))
        col0 = str(col)+'_encoding'
        #(1), 计算不良率，并使用不良率对原始值进行编码
        encoding_result = BadRateEncoding(trainData, col, 'target')
        trainData[col0], br_encoding = encoding_result['encoding'],encoding_result['br_rate']
        #(2), 将坏率编码值放入数值变量列表中
        numerical_var.append(col0)
        #(3), 保存编码结果，包括新列名和错误率
        encoded_features[col] = [col0, br_encoding]
            #(4), 删除原始值
        #del trainData[col]
        deleted_features.append(col)
    else:
        maxPcnt = MaximumBinPcnt(trainData, col)
        if maxPcnt > 0.9:
            print('{} is deleted because of large percentage of single bin'.format(col))
            deleted_features.append(col)
            categorical_var.remove(col)
            #del trainData[col]
            continue
        bad_bin = trainData.groupby([col])['target'].sum()
        if min(bad_bin) == 0:
            print('{} has 0 bad sample!'.format(col))
            col1 = str(col) + '_mergeByBadRate'
            #(1), 确定如何合并类别
            mergeBin = MergeBad0(trainData, col, 'target')
            #(2), 将原始数据转换为合并数据
            trainData[col1] = trainData[col].map(mergeBin)
            maxPcnt = MaximumBinPcnt(trainData, col1)
            if maxPcnt > 0.9:
                print('{} is deleted because of large percentage of single bin'.format(col))
                deleted_features.append(col)
                categorical_var.remove(col)
                del trainData[col]
                continue
            #(3) 如果合并的数据满足需求，我们就保留它
            merged_features[col] = [col1, mergeBin]
            WOE_IV = CalcWOE(trainData, col1, 'target')
            var_WOE[col1] = WOE_IV['WOE']
            var_IV[col1] = WOE_IV['IV']
            #del trainData[col]
            deleted_features.append(col)
        else:
            WOE_IV = CalcWOE(trainData, col, 'target')
            var_WOE[col] = WOE_IV['WOE']
            var_IV[col] = WOE_IV['IV']


'''
对于连续变量，我们做以下工作:
1、通过ChiMerge分割变量(默认情况下分成5个箱子)
2、检查坏率，如果不是单调的，我们减少箱子的数量，直到坏率单调
3、如果maximum bin占用超过90%，则删除该变量
'''
var_cutoff = {}
for col in numerical_var:
    print("{} is in processing".format(col))
    col1 = str(col) + '_Bin'
    #(1), 分割连续变量并保存截止点。特殊的，-1是一个特殊的情况，我们把它分成一个组
    if -1 in set(trainData[col]):
        special_attribute = [-1]
    else:
        special_attribute = []
    # 卡方分箱，返回分箱点
    cutOffPoints = ChiMerge_MaxInterval(trainData, col, 'target',special_attribute=special_attribute)
    var_cutoff[col] = cutOffPoints
    # 设置使得分箱覆盖所有训练样本外可能存在的值
    trainData[col1] = trainData[col].map(lambda x: AssignBin(x, cutOffPoints,special_attribute=special_attribute))

    #(2), 检查不良率是否为单调
    BRM = BadRateMonotone(trainData, col1, 'target',special_attribute=special_attribute)
    # 如果不单调就减少最大分箱数，进行重新分箱，再判断，直至bins=2或者bad rate单调
    if not BRM:
        for bins in range(4,1,-1):
            cutOffPoints = ChiMerge_MaxInterval(trainData, col, 'target',max_interval = bins,special_attribute=special_attribute)
            trainData[col1] = trainData[col].map(lambda x: AssignBin(x, cutOffPoints,special_attribute=special_attribute))
            BRM = BadRateMonotone(trainData, col1, 'target',special_attribute=special_attribute)
            if BRM:
                break
        var_cutoff[col] = cutOffPoints

    #(3), 检查是否有单个仓库占用总数的90%以上
    maxPcnt = MaximumBinPcnt(trainData, col1)
    if maxPcnt > 0.9:
        #del trainData[col1]
        deleted_features.append(col)
        numerical_var.remove(col)
        print('we delete {} because the maximum bin occupies more than 90%'.format(col))
        continue
    WOE_IV = CalcWOE(trainData, col1, 'target')
    var_IV[col] = WOE_IV['IV']
    var_WOE[col] = WOE_IV['WOE']
    #del trainData[col]

trainData.to_csv('C://Users//glf//Desktop//two//第六讲数据//allData_2a.csv', header=True,encoding='gbk', columns = trainData.columns, index=False)

# filewrite = open(path+'/数据/bank default/var_WOE.pkl','w')
filewrite = open('C://Users//glf//Desktop//two//第六讲数据//var_WOE.pkl','w')
pickle.dump(var_WOE, filewrite)
filewrite.close()


# filewrite = open(path+'/数据/bank default/var_IV.pkl','w')
filewrite = open('C://Users//glf//Desktop//two//第六讲数据//var_IV.pkl','w')
pickle.dump(var_IV, filewrite)
filewrite.close()


###########################################
# Step 4: 选择带有IV > 0.02的变量并分配WOE#
###########################################
trainData = pd.read_csv('C://Users//glf//Desktop//two//第六讲数据//allData_2a.csv', header=0, encoding='gbk')

num2str = ['SocialNetwork_13','SocialNetwork_12','UserInfo_6','UserInfo_5','UserInfo_10','UserInfo_17','city_match']
for col in num2str:
    trainData[col] = trainData[col].map(lambda x: str(x))


for col in var_WOE.keys():
    print(col)
    col2 = str(col)+"_WOE"
    if col in var_cutoff.keys():
        cutOffPoints = var_cutoff[col]
        special_attribute = []
        if - 1 in cutOffPoints:
            special_attribute = [-1]
        binValue = trainData[col].map(lambda x: AssignBin(x, cutOffPoints,special_attribute=special_attribute))
        trainData[col2] = binValue.map(lambda x: var_WOE[col][x])
    else:
        trainData[col2] = trainData[col].map(lambda x: var_WOE[col][x])

trainData.to_csv('C://Users//glf//Desktop//two//第六讲数据//allData_3.csv', header=True,encoding='gbk', columns = trainData.columns, index=False)

### (i) 选择IV高于阈值的特征
iv_threshould = 0.02
varByIV = [k for k, v in var_IV.items() if v > iv_threshould]


### (ii) (i)后任意一对特征与WOE共线性检验

var_IV_selected = {k:var_IV[k] for k in varByIV}
var_IV_sorted = sorted(var_IV_selected.iteritems(), key=lambda d:d[1], reverse = True)
var_IV_sorted = [i[0] for i in var_IV_sorted]

removed_var  = []
roh_thresould = 0.6
for i in range(len(var_IV_sorted)-1):
    if var_IV_sorted[i] not in removed_var:
        x1 = var_IV_sorted[i]+"_WOE"
        for j in range(i+1,len(var_IV_sorted)):
            if var_IV_sorted[j] not in removed_var:
                x2 = var_IV_sorted[j] + "_WOE"
                roh = np.corrcoef([trainData[x1], trainData[x2]])[0, 1]
                if abs(roh) >= roh_thresould:
                    print('the correlation coeffient between {0} and {1} is {2}'.format(x1, x2, str(roh)))
                    if var_IV[var_IV_sorted[i]] > var_IV[var_IV_sorted[j]]:
                        removed_var.append(var_IV_sorted[j])
                    else:
                        removed_var.append(var_IV_sorted[i])

var_IV_sortet_2 = [i for i in var_IV_sorted if i not in removed_var]

### (iii) 根据VIF > 10检查多共线
for i in range(len(var_IV_sortet_2)):
    x0 = trainData[var_IV_sortet_2[i]+'_WOE']
    x0 = np.array(x0)
    X_Col = [k+'_WOE' for k in var_IV_sortet_2 if k != var_IV_sortet_2[i]]
    X = trainData[X_Col]
    X = np.matrix(X)
    regr = LinearRegression()
    clr= regr.fit(X, x0)
    x_pred = clr.predict(X)
    R2 = 1 - ((x_pred - x0) ** 2).sum() / ((x0 - x0.mean()) ** 2).sum()
    vif = 1/(1-R2)
    if vif > 10:
        print("Warning: the vif for {0} is {1}".format(var_IV_sortet_2[i], vif))


###################################################################
# Step 5: 通过单因素分析和多元分析，对所选变量进行logistic回归分析#
###################################################################

### (1) 将单、多元分析后的所有特征进行logistic回归分析
var_WOE_list = [i+'_WOE' for i in var_IV_sortet_2]
y = trainData['target']
X = trainData[var_WOE_list]
X['intercept'] = [1]*X.shape[0]


LR = sm.Logit(y, X).fit()
summary = LR.summary()
pvals = LR.pvalues
pvals = pvals.to_dict()

### 有些特征不显著，所以我们需要逐个删除特征。
varLargeP = {k: v for k,v in pvals.items() if v >= 0.1}
varLargeP = sorted(varLargeP.iteritems(), key=lambda d:d[1], reverse = True)
while(len(varLargeP) > 0 and len(var_WOE_list) > 0):
    # 在每次迭代中，我们删除最不重要的特性并再次构建回归，直到
    # (1) 所有的特征都是显著的或
    # (2) 没有需要选择的特性
    varMaxP = varLargeP[0][0]
    if varMaxP == 'intercept':
        print('the intercept is not significant!')
        break
    var_WOE_list.remove(varMaxP)
    y = trainData['target']
    X = trainData[var_WOE]
    X['intercept'] = [1] * X.shape[0]

    LR = sm.Logit(y, X).fit()
    summary = LR.summary()
    pvals = LR.pvalues
    pvals = pvals.to_dict()
    varLargeP = {k: v for k, v in pvals.items() if v >= 0.1}
    varLargeP = sorted(varLargeP.iteritems(), key=lambda d: d[1], reverse=True)

'''
现在所有的特征都是显著的系数的符号是负的
var_WOE_list = ['UserInfo_15_encoding_WOE', u'ThirdParty_Info_Period6_10_WOE', u'ThirdParty_Info_Period5_2_WOE', 'UserInfo_16_encoding_WOE', 'WeblogInfo_20_encoding_WOE',
            'UserInfo_7_encoding_WOE', u'UserInfo_17_WOE', u'ThirdParty_Info_Period3_10_WOE', u'ThirdParty_Info_Period1_10_WOE', 'WeblogInfo_2_encoding_WOE',
            'UserInfo_1_encoding_WOE']
'''

saveModel =open(path+'/数据/bank default/LR_Model_Normal.pkl','w')
pickle.dump(LR,saveModel)
saveModel.close()

################################################################################
# Step 6(a): 基于步骤5中给出的变量，使用lasso and weights based建立logistic回归#
################################################################################
### 使用交叉验证选择最佳的正则化参数
X = trainData[var_WOE_list]   #默认情况下LogisticRegressionCV()填充符合截距
X = np.matrix(X)
y = trainData['target']
y = np.array(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=0)
X_train.shape, y_train.shape

model_parameter = {}
for C_penalty in np.arange(0.005, 0.2,0.005):
    for bad_weight in range(2, 101, 2):
        LR_model_2 = LogisticRegressionCV(Cs=[C_penalty], penalty='l1', solver='liblinear', class_weight={1:bad_weight, 0:1})
        LR_model_2_fit = LR_model_2.fit(X_train,y_train)
        y_pred = LR_model_2_fit.predict_proba(X_test)[:,1]
        scorecard_result = pd.DataFrame({'prob':y_pred, 'target':y_test})
        performance = KS_AR(scorecard_result,'prob','target')
        KS = performance['KS']
        model_parameter[(C_penalty, bad_weight)] = KS

################################################
# Step 6(b): 根据RF特征的重要性建立logistic回归#
################################################
### 建立随机森林模型来估计每个特征的重要性
### 在这种情况下，我们在进行单一分析之前使用了带有WOE编码的原始特征

X = trainData[var_WOE_list]
X = np.matrix(X)
y = trainData['target']
y = np.array(y)

RFC = RandomForestClassifier()
RFC_Model = RFC.fit(X,y)
features_rfc = trainData[var_WOE_list].columns
featureImportance = {features_rfc[i]:RFC_Model.feature_importances_[i] for i in range(len(features_rfc))}
featureImportanceSorted = sorted(featureImportance.iteritems(),key=lambda x: x[1], reverse=True)
# 我们选择了前10个功能
features_selection = [k[0] for k in featureImportanceSorted[:10]]

y = trainData['target']
X = trainData[features_selection]
X['intercept'] = [1]*X.shape[0]


LR = sm.Logit(y, X).fit()
summary = LR.summary()
"""
                           Logit Regression Results
==============================================================================
Dep. Variable:                 target   No. Observations:                30000
Model:                          Logit   Df Residuals:                    29989
Method:                           MLE   Df Model:                           10
Date:                Wed, 26 Apr 2017   Pseudo R-squ.:                 0.05762
Time:                        19:26:13   Log-Likelihood:                -7407.3
converged:                       True   LL-Null:                       -7860.2
                                        LLR p-value:                3.620e-188
==================================================================================================
                                     coef    std err          z      P>|z|      [0.025      0.975]
--------------------------------------------------------------------------------------------------
UserInfo_1_encoding_WOE           -1.0433      0.135     -7.756      0.000      -1.307      -0.780
WeblogInfo_20_encoding_WOE        -0.9011      0.089    -10.100      0.000      -1.076      -0.726
UserInfo_15_encoding_WOE          -0.9184      0.069    -13.215      0.000      -1.055      -0.782
UserInfo_7_encoding_WOE           -0.9891      0.096    -10.299      0.000      -1.177      -0.801
UserInfo_16_encoding_WOE          -0.9492      0.099     -9.603      0.000      -1.143      -0.756
ThirdParty_Info_Period1_10_WOE    -0.5942      0.143     -4.169      0.000      -0.874      -0.315
ThirdParty_Info_Period2_10_WOE    -0.0650      0.165     -0.395      0.693      -0.388       0.257
ThirdParty_Info_Period3_10_WOE    -0.2052      0.136     -1.511      0.131      -0.471       0.061
ThirdParty_Info_Period6_10_WOE    -0.6902      0.090     -7.682      0.000      -0.866      -0.514
ThirdParty_Info_Period5_10_WOE    -0.4018      0.100     -4.017      0.000      -0.598      -0.206
intercept                         -2.5382      0.024   -107.939      0.000      -2.584      -2.492
==================================================================================================
"""
