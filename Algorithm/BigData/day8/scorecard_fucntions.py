
### Calculate the cumulative frequences of events for each selected time window
def TimeWindowSelection(df, daysCol, time_windows):
    '''
    :param df: the dataset containg variabel of days
    :param daysCol: the column of days
    :param time_windows: the list of time window
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

def AssignBin(x, cutOffPoints):
    '''
    :param x: the value of variable
    :param cutOffPoints: the ChiMerge result for continous variable
    :return: bin number, indexing from 0
    for example, if cutOffPoints = [10,20,30], if x = 7, return Bin 0. If x = 35, return Bin 3
    '''
    numBin = len(cutOffPoints) + 1
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
    N = sum(regroup['total'])
    B = sum(regroup['bad'])
    regroup['good'] = regroup['total'] - regroup['bad']
    G = N - B
    regroup['bad_pcnt'] = regroup['bad'].map(lambda x: x*1.0/B)
    regroup['good_pcnt'] = regroup['good'].map(lambda x: x * 1.0 / G)
    regroup['WOE'] = regroup.apply(lambda x: np.log(x.good_pcnt*1.0/x.bad_pcnt),axis = 1)
    WOE_dict = regroup[[col,'WOE']].set_index(col).to_dict(orient='index')
    IV = regroup.apply(lambda x: (x.good_pcnt-x.bad_pcnt)*np.log(x.good_pcnt*1.0/x.bad_pcnt),axis = 1)
    IV = sum(IV)
    return {"WOE": WOE_dict, 'IV':IV}


def BadRateEncoding(df, col, target):
    '''
    :param df: dataframe containing feature and target
    :param col: the feature that needs to be encoded with bad rate, usually categorical type
    :param target: good/bad indicator
    :return: the assigned bad rate to encode the categorical fature
    '''
    total = df.groupby([col])[target].count()
    total = pd.DataFrame({'total': total})
    bad = df.groupby([col])[target].sum()
    bad = pd.DataFrame({'bad': bad})
    regroup = total.merge(bad, left_index=True, right_index=True, how='left')
    regroup.reset_index(level=0, inplace=True)
    regroup['bad_rate'] = regroup.apply(lambda x: x.bad*1.0/x.total,axis = 1)
    br_dict = regroup[[col,'bad_rate']].set_index([col]).to_dict(orient='index')
    badRateEnconding = df[col].map(lambda x: br_dict[x]['bad_rate'])
    return {'encoding':badRateEnconding, 'br_rate':br_dict}



def Chi2(df, total_col, bad_col, overallRate):
    '''
    :param df: the dataset containing the total count and bad count
    :param total_col: total count of each value in the variable
    :param bad_col: bad count of each value in the variable
    :param overallRate: the overall bad rate of the training set
    :return: the chi-square value
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



### ChiMerge_MaxInterval: split the continuous variable using Chi-square value by specifying the max number of intervals
def ChiMerge_MaxInterval_Original(df, col, target, max_interval = 5):
    '''
    :param df: the dataframe containing splitted column, and target column with 1-0
    :param col: splitted column
    :param target: target column with 1-0
    :param max_interval: the maximum number of intervals. If the raw column has attributes less than this parameter, the function will not work
    :return: the combined bins
    '''
    colLevels = set(df[col])
    # since we always combined the neighbours of intervals, we need to sort the attributes
    colLevels = sorted(list(colLevels))
    N_distinct = len(colLevels)
    if N_distinct <= max_interval:  #If the raw column has attributes less than this parameter, the function will not work
        print("The number of original levels for {} is less than or equal to max intervals".format(col))
        return colLevels[:-1]
    else:
        #Step 1: group the dataset by col and work out the total count & bad count in each level of the raw column
        total = df.groupby([col])[target].count()
        total = pd.DataFrame({'total':total})
        bad = df.groupby([col])[target].sum()
        bad = pd.DataFrame({'bad':bad})
        regroup =  total.merge(bad,left_index=True,right_index=True, how='left')
        regroup.reset_index(level=0, inplace=True)
        N = sum(regroup['total'])
        B = sum(regroup['bad'])
        #the overall bad rate will be used in calculating expected bad count
        overallRate = B*1.0/N
        # initially, each single attribute forms a single interval
        groupIntervals = [[i] for i in colLevels]
        groupNum = len(groupIntervals)
        while(len(groupIntervals)>max_interval):   #the termination condition: the number of intervals is equal to the pre-specified threshold
            # in each step of iteration, we calcualte the chi-square value of each atttribute
            chisqList = []
            for interval in groupIntervals:
                df2 = regroup.loc[regroup[col].isin(interval)]
                chisq = Chi2(df2, 'total','bad',overallRate)
                chisqList.append(chisq)
            #find the interval corresponding to minimum chi-square, and combine with the neighbore with smaller chi-square
            min_position = chisqList.index(min(chisqList))
            if min_position == 0:
                combinedPosition = 1
            elif min_position == groupNum - 1:
                combinedPosition = min_position -1
            else:
                if chisqList[min_position - 1]<=chisqList[min_position + 1]:
                    combinedPosition = min_position - 1
                else:
                    combinedPosition = min_position + 1
            groupIntervals[min_position] = groupIntervals[min_position]+groupIntervals[combinedPosition]
            # after combining two intervals, we need to remove one of them
            groupIntervals.remove(groupIntervals[combinedPosition])
            groupNum = len(groupIntervals)
        groupIntervals = [sorted(i) for i in groupIntervals]
        cutOffPoints = [i[-1] for i in groupIntervals[:-1]]
        return cutOffPoints


### ChiMerge_MaxInterval: split the continuous variable using Chi-square value by specifying the max number of intervals
def ChiMerge_MaxInterval(df, col, target, max_interval=5):
    '''
    :param df: the dataframe containing splitted column, and target column with 1-0
    :param col: splitted column
    :param target: target column with 1-0
    :param max_interval: the maximum number of intervals. If the raw column has attributes less than this parameter, the function will not work
    :return: the combined bins
    '''
    colLevels = sorted(list(set(df[col])))
    N_distinct = len(colLevels)
    if N_distinct <= max_interval:  # If the raw column has attributes less than this parameter, the function will not work
        print("The number of original levels for {} is less than or equal to max intervals".format(col))
        return colLevels[:-1]
    else:
        # Step 1: group the dataset by col and work out the total count & bad count in each level of the raw column
        if N_distinct > 100:
            ind_x = [int(i / 100.0 * N_distinct) for i in range(1, 100)]
            split_x = [colLevels[i] for i in ind_x]
            df['temp'] = df[col].map(lambda x: AssignGroup(x, split_x))
        else:
            df['temp'] = df[col]
        total = df.groupby(['temp'])[target].count()
        total = pd.DataFrame({'total': total})
        bad = df.groupby(['temp'])[target].sum()
        bad = pd.DataFrame({'bad': bad})
        regroup = total.merge(bad, left_index=True, right_index=True, how='left')
        regroup.reset_index(level=0, inplace=True)
        N = sum(regroup['total'])
        B = sum(regroup['bad'])
        # the overall bad rate will be used in calculating expected bad count
        overallRate = B * 1.0 / N
        # initially, each single attribute forms a single interval
        # since we always combined the neighbours of intervals, we need to sort the attributes
        colLevels = sorted(list(set(df['temp'])))
        groupIntervals = [[i] for i in colLevels]
        groupNum = len(groupIntervals)
        while (len(groupIntervals) > max_interval):  # the termination condition: the number of intervals is equal to the pre-specified threshold
            # in each step of iteration, we calcualte the chi-square value of each atttribute
            chisqList = []
            for interval in groupIntervals:
                df2 = regroup.loc[regroup['temp'].isin(interval)]
                chisq = Chi2(df2, 'total', 'bad', overallRate)
                chisqList.append(chisq)
            # find the interval corresponding to minimum chi-square, and combine with the neighbore with smaller chi-square
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
            # after combining two intervals, we need to remove one of them
            groupIntervals.remove(groupIntervals[combinedPosition])
            groupNum = len(groupIntervals)
        groupIntervals = [sorted(i) for i in groupIntervals]
        cutOffPoints = [i[-1] for i in groupIntervals[:-1]]
        del df['temp']
        return cutOffPoints

## determine whether the bad rate is monotone along the sortByVar
def BadRateMonotone(df, sortByVar, target):
    df2 = df.sort([sortByVar])
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

### If we find any categories with 0 bad, then we combine these categories with that having smallest non-zero bad rate
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