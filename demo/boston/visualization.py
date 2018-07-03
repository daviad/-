#!/usr/bin/env python3
#-*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# import seaborn as sns

from sklearn.datasets import load_boston

# pd.set_option('display.height',1000)
# pd.set_option('display.max_rows',500)
# pd.set_option('display.max_columns',500)
pd.set_option('display.width',1000)

#波士顿房价数据
boston=load_boston()
x=boston.data
y=boston.target
names=boston.feature_names 
print("feature_names:\n",names);
# print(boston.DESCR)
# print(boston)
# print(boston.keys())


df = pd.DataFrame(boston.data,columns=[boston.feature_names])
df['price'] = y
print ("describe:\n",df.describe())






# print(data.info)

# plt.scatter(x,y)
# plt.show()


