# 十分钟搞定pandas https://www.cnblogs.com/chaosimple/p/4153083.html

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 创建对象 
# 1、可以通过传递一个list对象来创建一个Series，pandas会默认创建整型索引：
s = pd.Series([1,3,5,np.nan,6,8])
print(type(s))

# concat
# 相同列合并到一起，不同时新建列，缺省值为NaN

df1 =pd.DataFrame({'a':[1,2,3],'b':[4,5,6]})
df2 =pd.DataFrame({'a':['a','b','c'],'c':['e','f','g']})

df = pd.concat([df1,df2])

print('df1:\n{df1} \ndf2:\n{df2}\ndf:\n{df}'.format(df1=df1,df2=df2,df=df))

# reset_index
df.reset_index(inplace=True)
print(df)

# drop
df.drop('index', axis=1,inplace=True)
print(df)

# reindex_axis
df =df.reindex_axis(df1.columns, axis=1)
print(df)