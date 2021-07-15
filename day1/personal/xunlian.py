import pandas as pd
import numpy as np
import missingno
import warnings
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_val_score

# pd.set_option('display.max_columns', None)
# pd.set_option('display.unicode.ambiguous_as_wide', True)
# pd.set_option('display.unicode.east_asian_width', True)
#消除警告
warnings.filterwarnings("ignore")
df = pd.read_csv('C:\\Users\\GUOLONGFEI\\Desktop\\wdbc.csv',header=None)
#缺失值可视化
#missingno.matrix(df)
#plt.show()
print(df.head())
#先将数据分为特征列和目标列
# x = df.loc[:,2:].values
# y = df.loc[:,1].values
# le = LabelEncoder()
# y = le.fit_transform(y)
# print(le.transform(['M','B']))
# print(pd.Series.value_counts(y))
#将数据分为训练集和测试集
# X_train,X_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=1)
# #接下来利用Pipeline类来进行工作流
# pipe_lr = Pipeline([('scl',StandardScaler()),('pca',PCA(n_components=2)),('clf',LogisticRegression(random_state=1))])
# pipe_lr.fit(X_train,y_train)
# #print("Test Accuracy:%.3f"%pipe_lr.score(x_test,y_test))
# #进行k折交叉验证
# kfold = StratifiedKFold(n_splits=10,random_state=1)
# scores=[]
# f1_scores=[]
# for k,(train,test) in enumerate(kfold.split(X_train,y_train)):
#     pipe_lr.fit(X_train[train],y_train[train])
#     p_pred = pipe_lr.predict(X_train[test])
#     f1_sc = f1_score(y_true= y_train[test],y_pred=p_pred)
#     f1_scores.append(f1_sc)
#     score = pipe_lr.score(X_train[test],y_train[test])
#     scores.append(score)
#     print('Fold: %s,Class dist.: %s,Acc: %.3f,score:%.3f' %(k+1,np.bincount(y_train[train]),score,f1_sc))
#计算f1值
# f1_scores = cross_val_score(estimator=pipe_lr,X=X_train,y=y_train,scoring='f1',cv=10,n_jobs=-1)
# print('CV f1 scores:%s'%f1_scores)
#计算AUC值
# aucs = cross_val_score(estimator=pipe_lr,X=X_train,y=y_train,scoring='roc_auc',cv=10,n_jobs=-1)
# print('CV auc scores:%s'%aucs)

