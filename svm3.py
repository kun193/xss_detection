# encoding=utf-8

import time

import numpy as np
import pandas as pd
#from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score,recall_score, precision_score,f1_score
from sklearn import datasets
from sklearn import svm
from sklearn.model_selection import train_test_split
if __name__ == '__main__':

    print('prepare datasets...')
    # Iris数据集
    # iris=datasets.load_iris()
    # features=iris.data
    # labels=iris.target

    # MINST数据集
    raw_data = pd.read_csv('file\\train_binary.csv', header=1)  # 读取csv数据，并将第一行视为表头，返回DataFrame类型
    data = raw_data.values
    features = data[::, 1::]
    labels = data[::, 0]    # 选取33%数据作为测试集，剩余为训练集

    train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size=0.3, random_state=0)

    time_2=time.time()
    print('Start training...')
    clf = svm.SVC()  # svm class
    clf.fit(train_features, train_labels)  # training the svc model
    time_3 = time.time()
    print('training cost %f seconds' % (time_3 - time_2))

    print('Start testing...')
    test_predict=clf.predict(test_features)
    time_4 = time.time()
    print('predicting cost %f seconds' % (time_4 - time_3))

    acc = accuracy_score(test_labels, test_predict)
    precision = precision_score(test_labels, test_predict)
    recall = recall_score(test_labels, test_predict)
    F1 = f1_score(test_labels, test_predict)
    # print("The accruacy score is %f" % acc)
    print("Accuracy score is ：", acc)
    print("Precision score is :", precision)
    print("Recall score is :", recall)
    print("F1 score is :", F1)

