import os
import numpy as np
import pandas as pd
import math
import pdb
from numpy import array
from numpy import shape
from numpy import ones
from numpy import random

def sigmoid(inX):
    return 1/(1+np.exp(-inX))

def classifyVector(inX,trainWeights):

    prob=sigmoid(sum(inX*trainWeights))
    if prob>0.5:return 1
    else : return 0

def stocGradAscent1(dataMatrix,classLabels,numIter=150):
    m = shape(dataMatrix)[0]
    weights=ones(1)
    for j in range(numIter):
        dataIndex=list(range(m))
        for i in range(m):
            alpha=4/(1+i+j)+0.01
            randIndex=int(random.uniform(0,len(dataIndex)-1))
            h=sigmoid(sum(dataMatrix[randIndex] * weights))
            error=classLabels[randIndex]-h
            weights=weights + alpha*dataMatrix[randIndex]*error
            del(dataIndex[randIndex])
    return weights


def trainTestSplit(X,test_size=0.3):
    X_num=X.shape[0]
    train_index=list(range(X_num))
    test_index=[]
    test_num=int(X_num*test_size)
    for i in range(test_num):
        randomIndex=int(np.random.uniform(0,len(train_index)))
        test_index.append(train_index[randomIndex])
        del train_index[randomIndex]
    train=X.ix[train_index]
    test=X.ix[test_index]
    return train,test

def colicTest(tv,tu):
    frTrain,frTest = trainTestSplit(tv,test_size=0.3)
    train_data = np.array(frTrain[tu])
    train_x_list = train_data.tolist()
    label_data = np.array(frTrain['LeavePct'])
    label_x_list = label_data.tolist()

    trainSet = np.concatenate((array([1]), array(train_x_list)))
    trainLabels=label_x_list

    trainWeights=stocGradAscent1(array(trainSet),trainLabels,500)
    errorCount=0
    numTestVec=frTest.shape[0]
    test_data = np.array(frTest[tu])
    test_x_list = test_data.tolist()
    label_data1 = np.array(frTest['LeavePct'])
    label_y_list = label_data1.tolist()
    j=-1
    for i in test_x_list:
        j=j+1
        lineArr =np.concatenate((array([1]),array([i])))
        if classifyVector(array(lineArr),trainWeights)!= label_y_list[j]:
            errorCount+=1
    errorRate=(float(errorCount)/numTestVec)
    return errorRate


dirname = os.getcwd()
filename = 'UKIP.csv'
df0 = pd.DataFrame.from_csv(os.path.join(dirname, filename))

binominal = lambda x: round(x/100)
d1 = df0['LeavePct'].apply(binominal)
df1 = df0.drop('LeavePct', axis=1)

df1['LeavePct'] = d1
column_headers = list(df1.columns.values)

indicator = column_headers[0]
numTests = 10;
errorSum = 0
for k in range(numTests):
    errorSum+=colicTest(df1, indicator)
print(indicator + ':' + str(errorSum/float(numTests)))