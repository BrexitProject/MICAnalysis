import os
import numpy as np
import pandas as pd
import math
import pdb
from sklearn import svm
from sklearn.model_selection import  GridSearchCV,train_test_split
from sklearn.externals import joblib

dirname = os.getcwd()
filename = 'No_religion.csv'
df0 = pd.DataFrame.from_csv(os.path.join(dirname, filename))

binominal = lambda x: round(x/100)
d1 = df0['LeavePct'].apply(binominal)
df1 = df0.drop('LeavePct', axis=1)

df1['LeavePct'] = d1
column_headers = list(df1.columns.values)


i = column_headers[0]
if i != 'LeavePct':
    x=df1.loc[df1[i] > 0, i].values

    y = df1.loc[df1[i] > 0, 'LeavePct'].values


x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7)

if (y_train == 1).all() or (y_train == 0).all():
    pass
else:
    tuned_parameters = {
        'kernel': ['rbf'],
        'C': [0.25, 0.5, 1, 2, 4, 8],
        'gamma': [0.125, 0.25, 0.5, 1, 2, 4]
    }
    clf = GridSearchCV(svm.SVC(), tuned_parameters, cv=3)
    clf.fit(x_train.reshape(-1, 1), y_train.reshape(-1, 1))
    print(clf.best_estimator_)


        # save model
    joblib.dump(clf, 'D:/svm/models/svm_train_No_religion')
        # load model from file
    clf = joblib.load('D:/svm/models/svm_train_No_religion')

    print(clf.score(x_train.reshape(-1, 1), y_train.reshape(-1, 1)))  # 精度
    y_hat = clf.predict(x_train.reshape(-1, 1))
    acc = clf.score(x_test.reshape(-1, 1), y_test.reshape(-1, 1))

    print(' %s accuracy on test set: %f' % (i,acc))





