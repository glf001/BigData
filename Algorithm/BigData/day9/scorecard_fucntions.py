import pandas as pd
import numpy as np
import random

#计算每个选定时间窗口的事件累积频率
def TimeWindowSelection(df, daysCol, time_windows):
    '''
    :param df: the dataset containg variabel of days   包含天数变量的数据集
    :param daysCol: the column of days                 天数的一列
    :param time_windows: 时间窗口列表
    :return:
    '''
    freq_tw = {}
    for tw in time_windows:
        freq = sum(df[daysCol].apply(lambda x: int(x<=tw)))
        freq_tw[tw] = freq
    return freq_tw


def ChangeContent(x):
    y = x.upper()
    if y == '_MOBILEPHONE':
        y = '_PHONE'
    return y

def MissingCategorial(df,x):
    missing_vals = df[x].map(lambda x: int(x!=x))
    return sum(missing_vals)*1.0/df.shape[0]

def MissingContinuous(df,x):
    missing_vals = df[x].map(lambda x: int(np.isnan(x)))
    return sum(missing_vals) * 1.0 / df.shape[0]

def MakeupRandom(x, sampledList):
    if x==x:
        return x
    else:
        randIndex = random.randint(0, len(sampledList)-1)
        return sampledList[randIndex]

def AssignBin(x, cutOffPoints,special_attribute=[]):
    '''
    :param x: the value of variable                   变量的值
    :param cutOffPoints: the ChiMerge result for continous variable    连续变量的嵌合结果
    :param special_attribute:  the special attribute which should be assigned separately      应该单独分配的特殊属性
    :return: bin number, indexing from 0
    for example, if cutOffPoints = [10,20,30], if x = 7, return Bin 0. If x = 35, return Bin 3
    '''
    numBin = len(cutOffPoints) + 1 + len(special_attribute)
    if x in special_attribute:
        i = special_attribute.index(x)+1
        return 'Bin {}'.format(0-i)
    if x<=cutOffPoints[0]:
        return 'Bin 0'
    elif x > cutOffPoints[-1]:
        return 'Bin {}'.format(numBin-1)
    else:
        for i in range(0,numBin-1):
            if cutOffPoints[i] < x <=  cutOffPoints[i+1]:
                return 'Bin {}'.format(i+1)


def MaximumBinPcnt(df,col):
    N = df.shape[0]
    total = df.groupby([col])[col].count()
    pcnt = total*1.0/N
    return max(pcnt)

def CalcWOE(df, col, target):
    '''
    :param df: dataframe containing feature and target          包含特性和目标的数据帧
    :param col: the feature that needs to be calculated the WOE and iv, usually categorical type   需要计算的特征，WOE和iv，通常是分类类型
    :param target: good/bad indicator       好的/坏的指标
    :return: WOE and IV in a dictionary     一本字典里的WOE和IV
    '''
    total = df.groupby([col])[target].count()
    total = pd.DataFrame({'total': total})
    bad = df.groupby([col])[target].sum()
    bad = pd.DataFrame({'bad': bad})
    regroup = total.merge(bad, left_index=True, right_index=True, how='left')
    regroup.reset_index(level=0, inplace=True)
    N = sum(regroup['total'])
    B = sum(regroup['bad'])
    regroup['good'] = regroup['total'] - regroup['bad']
    G = N - B
    regroup['bad_pcnt'] = regroup['bad'].map(lambda x: x*1.0/B)
    regroup['good_pcnt'] = regroup['good'].map(lambda x: x * 1.0 / G)
    regroup['WOE'] = regroup.apply(lambda x: np.log(x.good_pcnt*1.0/x.bad_pcnt),axis = 1)
    WOE_dict = regroup[[col,'WOE']].set_index(col).to_dict(orient='index')
    for k, v in WOE_dict.items():
        WOE_dict[k] = v['WOE']
    IV = regroup.apply(lambda x: (x.good_pcnt-x.bad_pcnt)*np.log(x.good_pcnt*1.0/x.bad_pcnt),axis = 1)
    IV = sum(IV)
    return {"WOE": WOE_dict, 'IV':IV}


def BadRateEncoding(df, col, target):
    '''
    :param df: dataframe containing feature and target    包含特性和目标的数据帧
    :param col: the feature that needs to be encoded with bad rate, usually categorical type     需要用不良率编码的特性，通常是分类类型
    :param target: good/bad indicator
    :return: the assigned bad rate to encode the categorical feature    编码分类特征所指定的不良率
    '''
    total = df.groupby([col])[target].count()
    total = pd.DataFrame({'total': total})
    bad = df.groupby([col])[target].sum()
    bad = pd.DataFrame({'bad': bad})
    regroup = total.merge(bad, left_index=True, right_index=True, how='left')
    regroup.reset_index(level=0, inplace=True)
    regroup['bad_rate'] = regroup.apply(lambda x: x.bad*1.0/x.total,axis = 1)
    br_dict = regroup[[col,'bad_rate']].set_index([col]).to_dict(orient='index')
    for k, v in br_dict.items():
        br_dict[k] = v['bad_rate']
    badRateEnconding = df[col].map(lambda x: br_dict[x])
    return {'encoding':badRateEnconding, 'br_rate':br_dict}



def Chi2(df, total_col, bad_col, overallRate):
    '''
    :param df: the dataset containing the total count and bad count   包含总计数和坏计数的数据集
    :param total_col: total count of each value in the variable      变量中每个值的总数
    :param bad_col: bad count of each value in the variable         变量中每个值的坏计数
    :param overallRate: the overall bad rate of the training set   训练集的整体不良率
    :return: the chi-square value                                 卡方值
    '''
    df2 = df.copy()
    df2['expected'] = df[total_col].apply(lambda x: x*overallRate)
    combined = zip(df2['expected'], df2[bad_col])
    chi = [(i[0]-i[1])**2/i[0] for i in combined]
    chi2 = sum(chi)
    return chi2

def AssignGroup(x, bin):
    N = len(bin)
    if x<=min(bin):
        return min(bin)
    elif x>max(bin):
        return 10e10
    else:
        for i in range(N-1):
            if bin[i] < x <= bin[i+1]:
                return bin[i+1]


#ChiMerge_MaxInterval:通过指定最大间隔数，使用卡方值分割连续变量
def ChiMerge_MaxInterval(df, col, target, max_interval=5,special_attribute=[]):
    '''
    :param df: the dataframe containing splitted column, and target column with 1-0     包含拆分列的数据帧，以及1-0的目标列
    :param col: splitted column                                                         结论根据列
    :param target: target column with 1-0                                              目标列为1-0
    :param max_interval: the maximum number of intervals. If the raw column has attributes less than this parameter, the function will not work   最大间隔数。如果原始列的属性小于此参数，则该函数将无法工作
    :return: the combined bins                                                        合并后的垃圾箱
    '''
    colLevels = sorted(list(set(df[col])))
    N_distinct = len(colLevels)
    if N_distinct <= max_interval:  # 如果原始列的属性小于此参数，则该函数将无法工作
        print("The number of original levels for {} is less than or equal to max intervals".format(col))
        return colLevels[:-1]
    else:
        if len(special_attribute)>=1:
            df1 = df.loc[df[col].isin(special_attribute)]
            df2 = df.loc[~df[col].isin(special_attribute)]
        else:
            df2 = df.copy()
        N_distinct = len(list(set(df2[col])))
        # Step 1: 分组数据集的col和工作出总计数和坏计数在每一级的原始列
        if N_distinct > 100:
            ind_x = [int(i / 100.0 * N_distinct) for i in range(1, 100)]
            split_x = [colLevels[i] for i in ind_x]
            df2['temp'] = df2[col].map(lambda x: AssignGroup(x, split_x))
        else:
            df2['temp'] = df[col]
        total = df2.groupby(['temp'])[target].count()
        total = pd.DataFrame({'total': total})
        bad = df2.groupby(['temp'])[target].sum()
        bad = pd.DataFrame({'bad': bad})
        regroup = total.merge(bad, left_index=True, right_index=True, how='left')
        regroup.reset_index(level=0, inplace=True)
        # 总体不良率将用于计算预期不良计数
        # overallRate = B * 1.0 / N
        try:
            N = sum(regroup['total'])
            B = sum(regroup['bad'])
            overallRate = B * 1.0 / N
            colLevels = sorted(list(set(df2['temp'])))
            groupIntervals = [[i] for i in colLevels]
            groupNum = len(groupIntervals)
            # 最后的分割间隔应该是指定的最大间隔减去特殊属性的数量
            split_intervals = max_interval - len(special_attribute)
            while (len(groupIntervals) > split_intervals):  # 终止条件:间隔数等于预先设定的阈值
                # 在迭代的每一步，我们计算每个属性的卡方值
                chisqList = []
                for interval in groupIntervals:
                    df2b = regroup.loc[regroup['temp'].isin(interval)]
                    chisq = Chi2(df2b, 'total', 'bad', overallRate)
                    chisqList.append(chisq)
                # 找出卡方最小对应的区间，并结合卡方较小的邻域
                min_position = chisqList.index(min(chisqList))
                if min_position == 0:
                    combinedPosition = 1
                elif min_position == groupNum - 1:
                    combinedPosition = min_position - 1
                else:
                    if chisqList[min_position - 1] <= chisqList[min_position + 1]:
                        combinedPosition = min_position - 1
                    else:
                        combinedPosition = min_position + 1
                groupIntervals[min_position] = groupIntervals[min_position] + groupIntervals[combinedPosition]
                # 合并两个间隔后，我们需要删除其中一个
                groupIntervals.remove(groupIntervals[combinedPosition])
                groupNum = len(groupIntervals)
            groupIntervals = [sorted(i) for i in groupIntervals]
            cutOffPoints = [max(i) for i in groupIntervals[:-1]]
            cutOffPoints = special_attribute + cutOffPoints
        except:
            print('被除数不能是0!')
        finally:
            return cutOffPoints
        # if B == 0:
        #     print('error:')
        # overallRate = B * 1.0 / N
        # 最初，每个单独的属性构成一个单独的间隔
        # 因为我们总是合并间隔的邻居，所以我们需要对属性进行排序
        # colLevels = sorted(list(set(df2['temp'])))
        # groupIntervals = [[i] for i in colLevels]
        # groupNum = len(groupIntervals)
        # #最后的分割间隔应该是指定的最大间隔减去特殊属性的数量
        # split_intervals = max_interval - len(special_attribute)
        # while (len(groupIntervals) > split_intervals):  # 终止条件:间隔数等于预先设定的阈值
        #     # 在迭代的每一步，我们计算每个属性的卡方值
        #     chisqList = []
        #     for interval in groupIntervals:
        #         df2b = regroup.loc[regroup['temp'].isin(interval)]
        #         chisq = Chi2(df2b, 'total', 'bad', overallRate)
        #         chisqList.append(chisq)
        #     # 找出卡方最小对应的区间，并结合卡方较小的邻域
        #     min_position = chisqList.index(min(chisqList))
        #     if min_position == 0:
        #         combinedPosition = 1
        #     elif min_position == groupNum - 1:
        #         combinedPosition = min_position - 1
        #     else:
        #         if chisqList[min_position - 1] <= chisqList[min_position + 1]:
        #             combinedPosition = min_position - 1
        #         else:
        #             combinedPosition = min_position + 1
        #     groupIntervals[min_position] = groupIntervals[min_position] + groupIntervals[combinedPosition]
        #     # 合并两个间隔后，我们需要删除其中一个
        #     groupIntervals.remove(groupIntervals[combinedPosition])
        #     groupNum = len(groupIntervals)
        # groupIntervals = [sorted(i) for i in groupIntervals]
        # cutOffPoints = [max(i) for i in groupIntervals[:-1]]
        # cutOffPoints = special_attribute + cutOffPoints
        # return cutOffPoints

#确定坏率是否单调沿sortByVar
def BadRateMonotone(df, sortByVar, target,special_attribute = []):
    '''
    :param df: the dataset contains the column which should be monotone with the bad rate and bad column  数据集包含的列应该是单调的，有坏率和坏列
    :param sortByVar: the column which should be monotone with the bad rate    这一列应该是单调的与坏率
    :param target: the bad column                                             坏列
    :param special_attribute: some attributes should be excluded when checking monotone   在检查单调时，应该排除一些属性
    :return:
    '''
    df2 = df.loc[~df[sortByVar].isin(special_attribute)]
    df2 = df2.sort([sortByVar])
    total = df2.groupby([sortByVar])[target].count()
    total = pd.DataFrame({'total': total})
    bad = df2.groupby([sortByVar])[target].sum()
    bad = pd.DataFrame({'bad': bad})
    regroup = total.merge(bad, left_index=True, right_index=True, how='left')
    regroup.reset_index(level=0, inplace=True)
    combined = zip(regroup['total'],regroup['bad'])
    badRate = [x[1]*1.0/x[0] for x in combined]
    badRateMonotone = [badRate[i]<badRate[i+1] for i in range(len(badRate)-1)]
    Monotone = len(set(badRateMonotone))
    if Monotone == 1:
        return True
    else:
        return False

#如果我们发现任何有0坏的类别，我们就把这些类别和有最小非零坏率的类别结合起来
def MergeBad0(df,col,target):
    '''
     :param df: dataframe containing feature and target
     :param col: the feature that needs to be calculated the WOE and iv, usually categorical type
     :param target: good/bad indicator
     :return: WOE and IV in a dictionary
     '''
    total = df.groupby([col])[target].count()
    total = pd.DataFrame({'total': total})
    bad = df.groupby([col])[target].sum()
    bad = pd.DataFrame({'bad': bad})
    regroup = total.merge(bad, left_index=True, right_index=True, how='left')
    regroup.reset_index(level=0, inplace=True)
    regroup['bad_rate'] = regroup.apply(lambda x: x.bad*1.0/x.total,axis = 1)
    regroup = regroup.sort_values(by = 'bad_rate')
    col_regroup = [[i] for i in regroup[col]]
    for i in range(regroup.shape[0]):
        col_regroup[1] = col_regroup[0] + col_regroup[1]
        col_regroup.pop(0)
        if regroup['bad_rate'][i+1] > 0:
            break
    newGroup = {}
    for i in range(len(col_regroup)):
        for g2 in col_regroup[i]:
            newGroup[g2] = 'Bin '+str(i)
    return newGroup

# Calculate the KS and AR for the socrecard model     计算板卡模型的KS和AR
def KS_AR(df, score, target):
    '''
    :param df: the dataset containing probability and bad indicator    包含概率和坏指标的数据集
    :param score:
    :param target:
    :return:
    '''
    total = df.groupby([score])[target].count()
    bad = df.groupby([score])[target].sum()
    all = pd.DataFrame({'total':total, 'bad':bad})
    all['good'] = all['total'] - all['bad']
    all[score] = all.index
    all = all.sort_values(by=score,ascending=False)
    all.index = range(len(all))
    all['badCumRate'] = all['bad'].cumsum() / all['bad'].sum()
    all['goodCumRate'] = all['good'].cumsum() / all['good'].sum()
    all['totalPcnt'] = all['total'] / all['total'].sum()
    arList = [0.5 * all.loc[0, 'badCumRate'] * all.loc[0, 'totalPcnt']]
    for j in range(1, len(all)):
        ar0 = 0.5 * sum(all.loc[j - 1:j, 'badCumRate']) * all.loc[j, 'totalPcnt']
        arList.append(ar0)
    arIndex = (2 * sum(arList) - 1) / (all['good'].sum() * 1.0 / all['total'].sum())
    KS = all.apply(lambda x: x.badCumRate - x.goodCumRate, axis=1)
    return {'AR':arIndex, 'KS': max(KS)}