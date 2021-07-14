import pandas as pd
from numpy import *
import tensorflow as tf
from tensorflow.contrib.learn.python.learn.estimators import SKCompat
from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt
from itertools import cycle
from sklearn.metrics import roc_curve, auc, roc_auc_score,precision_recall_curve, average_precision_score
import operator


modelData = pd.read_csv(path+'/数据/bank attrition/modelData.csv', header = 0)
allFeatures = list(modelData.columns)
#remove the class label and cust id from features
allFeatures.remove('CUST_ID')
allFeatures.remove('CHURN_CUST_IND')
modelData2 = modelData.copy()

#normalize the features using max-min to convert the values into [0,1] interval
def MaxMinNorm(df,col):
    ma, mi = max(df[col]), min(df[col])
    rangeVal = ma - mi
    if rangeVal == 0:
        print(col)
    df[col] = df[col].map(lambda x:(x-mi)*1.0/rangeVal)

for numFeatures in allFeatures:
    MaxMinNorm(modelData2, numFeatures)

#we need to remove columns of 'TELEBANK_ALL_TX_NUM', 'RATIO_21' since they are constants
allFeatures.remove('TELEBANK_ALL_TX_NUM')
allFeatures.remove('RATIO_21')


#split the dataset into training and testing parts
x_train, x_test, y_train, y_test = train_test_split(modelData2[allFeatures],modelData2['CHURN_CUST_IND'], test_size=0.4,random_state=9)

x_train = matrix(x_train)
y_train = matrix(y_train).T
x_test = matrix(x_test)
y_test = matrix(y_test).T

y_test = y_test.reshape(-1,).tolist()[0]


#numnber of input layer nodes: dimension =
#number of hidden layer & number of nodes in them: hidden_units
#full link or not: droput. dropout = 1 means full link
#activation function: activation_fn. By default it is relu
#learning rate:

#Example: select the best number of units in the 1-layer hidden layer
#model_dir = path can make the next iteration starting from last termination
#define the DNN with 1 hidden layer
no_hidden_units_selection = {}
feature_columns = [tf.contrib.layers.real_valued_column("", dimension = x_train.shape[1])]
for no_hidden_units in range(10,101,10):
    print("the current choise of hidden units number is {}".format(no_hidden_units))
    clf0 = tf.contrib.learn.DNNClassifier(feature_columns = feature_columns,
                                          #hidden_units = [no_hidden_units],
                                          hidden_units=[no_hidden_units, no_hidden_units+10],
                                          n_classes=2,
                                          dropout = 0.5
                                          #optimizer=tf.train.ProximalAdagradOptimizer(learning_rate=0.1,l1_regularization_strength=0.001
                                          #model_dir = path
                                          #learning_rate=0.1
                                          )
    clf = SKCompat(clf0)
    clf.fit(x_train, y_train, batch_size=256,steps = 100000)
    #monitor the performance of the model using AUC score
    #clf_pred = clf._estimator.predict(x_test)
    #y_pred = [i for i in clf_pred]
    clf_pred_proba = clf._estimator.predict_proba(x_test)
    pred_proba = [i[1] for i in clf_pred_proba]
    auc_score = roc_auc_score(y_test,pred_proba)
    no_hidden_units_selection[no_hidden_units] = auc_score
best_hidden_units = max(no_hidden_units_selection.iteritems(), key=operator.itemgetter(1))[0]


#Example: check the dropout effect
dropout_selection = {}
feature_columns = [tf.contrib.layers.real_valued_column("", dimension = x_train.shape[1])]
for dropout_prob in linspace(0,0.99,100):
    print("the current choise of drop out rate is {}".format(dropout_prob))
    clf0 = tf.contrib.learn.DNNClassifier(feature_columns = feature_columns,
                                          hidden_units = [no_hidden_units],
                                          n_classes=2,
                                          dropout = dropout_prob
                                          #optimizer=tf.train.ProximalAdagradOptimizer(learning_rate=0.1,l1_regularization_strength=0.001
                                          #model_dir = path
                                          #learning_rate=0.1
                                          )
    clf = SKCompat(clf0)
    clf.fit(x_train, y_train, batch_size=256,steps = 10000)
    #monitor the performance of the model using AUC score
    #clf_pred = clf._estimator.predict(x_test)
    #y_pred = [i for i in clf_pred]
    clf_pred_proba = clf._estimator.predict_proba(x_test)
    pred_proba = [i[1] for i in clf_pred_proba]
    auc_score = roc_auc_score(y_test,pred_proba)
    dropout_selection[dropout_prob] = auc_score
best_dropout_prob = max(dropout_selection.iteritems(), key=operator.itemgetter(1))[0]
