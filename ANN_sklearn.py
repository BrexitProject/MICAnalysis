import sklearn
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import datasets
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
import numpy as np

df0 = pd.DataFrame.from_csv(r'D:\svm\data\Happiness.csv')

binominal = lambda x: round(x/100)
d1 = df0['LeavePct'].apply(binominal)
df1 = df0.drop('LeavePct', axis=1)

df1['LeavePct'] = d1

column_headers = list(df1.columns.values)
if column_headers[0] != 'LeavePct':
    x=df1.loc[df1[column_headers[0]] > 0, column_headers[0]].values
    y = df1.loc[df1[column_headers[0]] > 0, 'LeavePct'].values

sum=0
fill=0
for i in range(10):
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7)
    t1=x_train.reshape(-1, 1)
    t2=x_test.reshape(-1,1)

    model = MLPClassifier(activation='logistic', solver='lbfgs',hidden_layer_sizes=(10))  # 神经网络
    poly = PolynomialFeatures(3)
    tn1=poly.fit_transform(t1)
    tn2=poly.fit_transform(t2)
    model.fit(tn1, y_train.reshape(-1, 1))
    pdb.set_trace()
    print(model.coefs_)
    sum += model.score(tn2, y_test.reshape(-1,1))
    fill += model.score(tn1, y_train.reshape(-1,1))

print('train accuracy: {:.3f}'.format(fill/10))
print('test accuracy: {:.3f}'.format(sum/10))