#!/usr/bin/env python3 
# coding=utf-8

# Kaggle泰坦尼克预测(完整分析)
# https://blog.csdn.net/guoxinian/article/details/73740746

'''
网上存在大量的解决这个问题的方法好像都是相互转载的，其实需要区分是python3本身的问题还是工具的问题

1.先在终端输出中文，可以

2.用工具输出，报以上错误

so：工具问题

解决sublime text3 输出问题的方法：

修改Sublime Text3中的设置

Preferences > Browse Packages > User > Python3.sublime-build

如果不存在这个文件，可以通过新建编译系统保存为这个文件

加上一句 env ，文件内容如下：

{
  "cmd":["/usr/local/bin/python3","-u","$file"],
  "file_regex": "^[ ]*File \"(...*?)\", line ([0-9]*)",
  "selector": "source.python",
  "env": {"LANG": "en_US.UTF-8"}
}

问题完美解决，加油
'''

import sys
print(sys.stdout.encoding)
print("我")


import pandas as pd
import numpy as np

# pd.set_option('display.height',1000)
# pd.set_option('display.max_rows',500)
# pd.set_option('display.max_columns',500)
pd.set_option('display.width',1000)

data_train = pd.read_csv('./data/train.csv')
# print(data_train)
# print(data_train.info())

'''
通常遇到缺值的情况，我们会有几种常见的处理方式

如果缺值的样本占总数比例极高，我们可能就直接舍弃了，作为特征加入的话，可能反倒带入noise，影响最后的结果了
如果缺值的样本适中，而该属性非连续值特征属性(比如说类目属性)，那就把NaN作为一个新类别，加到类别特征中
如果缺值的样本适中，而该属性为连续值特征属性，有时候我们会考虑给定一个step(比如这里的age，我们可以考虑每隔2/3岁为一个步长)，然后把它离散化，之后把NaN作为一个type加到属性类目中。
有些情况下，缺失的值个数并不是特别多，那我们也可以试着根据已有的值，拟合一下数据，补充上。

'''

# 这里用scikit-learn中的RandomForest来拟合一下缺失的年龄数据(注：RandomForest是一个用在原始数据中做不同采样，建立多颗DecisionTree，再进行average等等来降低过拟合现象，提高结果的机器学习算法，我们之后会介绍到)
from sklearn.ensemble import RandomForestRegressor
### 使用 RandomForestClassifier 填补缺失的年龄属性
def set_missing_ages(df):

    # 把已有的数值型特征取出来丢进Random Forest Regressor中
    # Pandas 下标存取操作 https://www.jianshu.com/p/2900cfee565e
    age_df = df[['Age','Fare', 'Parch', 'SibSp', 'Pclass']]

    # 乘客分成已知年龄和未知年龄两部分
    known_age = age_df[age_df.Age.notnull()].as_matrix()
    unknown_age = age_df[age_df.Age.isnull()].as_matrix()

    # print(age_df[age_df.Age.notnull()])
    # print(known_age)
    # y即目标年龄
    y = known_age[:, 0]
    # print(y)
    # X即特征属性值
    X = known_age[:, 1:]

    # fit到RandomForestRegressor之中
    rfr = RandomForestRegressor(random_state=0, n_estimators=2000, n_jobs=-1)
    rfr.fit(X, y)

    # 用得到的模型进行未知年龄结果预测
    predictedAges = rfr.predict(unknown_age[:, 1::])

    # 用得到的预测结果填补原缺失数据
    df.loc[ (df.Age.isnull()), 'Age' ] = predictedAges 

    return df, rfr

def set_Cabin_type(df):
    df.loc[ (df.Cabin.notnull()), 'Cabin' ] = "Yes"
    df.loc[ (df.Cabin.isnull()), 'Cabin' ] = "No"
    return df

data_train, rfr = set_missing_ages(data_train)
data_train = set_Cabin_type(data_train)

data_train.loc[data_train['Sex'] == 'male','Sex'] = 0
data_train.loc[data_train['Sex'] == 'female','Sex'] = 1

data_train.loc[data_train['Cabin'] == 'Yes','Cabin'] = 1
data_train.loc[data_train['Cabin'] == 'No','Cabin'] = 0

data_train['Embarked'] = data_train['Embarked'].fillna('S')
data_train.loc[data_train['Embarked'] == 'S','Embarked'] = 0
data_train.loc[data_train['Embarked'] == 'C','Embarked'] = 1
data_train.loc[data_train['Embarked'] == 'Q','Embarked'] = 2

data_train.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1, inplace=True)


# print(data_train
# print(data_train.describe())
'''
如果大家了解逻辑回归与梯度下降的话，会知道，各属性值之间scale差距太大，将对收敛速度造成几万点伤害值！甚至不收敛！ (╬▔皿▔)…所以我们先用scikit-learn里面的preprocessing模块对这俩货做一个scaling，所谓scaling，其实就是将一些变化幅度较大的特征化到[-1,1]之内。
'''
import sklearn.preprocessing as preprocessing

def set_scale(df):
    scaler = preprocessing.StandardScaler()
    age_scale_param = scaler.fit(df['Age'].as_matrix())
    df['Age_scaled'] = scaler.fit_transform(df['Age'], age_scale_param)
    fare_scale_param = scaler.fit(df['Fare'])
    df['Fare_scaled'] = scaler.fit_transform(df['Fare'], fare_scale_param)
    return df

# set_scale(data_train)

print(data_train)

