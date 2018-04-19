#!/usr/bin/python
# -*- coding: UTF-8 -*-
# 文件名: logistic_regression.py
import random
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


def load_from_file(f,rate):
    #用于从txt文本中读取数据并划分训练集及测试集
    with open(f,'r') as fr:
        raw_data=fr.readlines()
    data = list(map(lambda x:x.strip().split(),raw_data))
    data_ = list(map(lambda x:list(map(lambda listx:int(listx.split(':')[-1]),x)),data))
    random.shuffle(data_)
    train_data = np.array(data_[:int(rate*len(data_))])
    test_data = np.array(data_[int(rate*len(data_)):])
    return train_data,test_data



train_data,test_data = load_from_file('./健康状况训练集.txt',0.7)
train_l = train_data[:,0]
train_f = train_data[:,1:]

# print(train_l)
# print(train_f)
lr =LogisticRegression(solver='lbfgs')
lr.fit(train_f,train_l)
# 根据训练集对模型进行训练

test_l = test_data[:,0]
test_f = test_data[:,1:]
# 整理测试集标签和特征
result = lr.predict(test_f)
# 使用训练好的模型对于测试集进行验证
wrong_list = list(zip(test_l,result))
a = list(map(lambda t:abs(t[0]-t[1]),wrong_list))
print(sum(a)/len(a))
print(accuracy_score(test_l,result))
