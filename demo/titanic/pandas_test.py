# 十分钟搞定pandas https://www.cnblogs.com/chaosimple/p/4153083.html

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 创建对象 
# 1、可以通过传递一个list对象来创建一个Series，pandas会默认创建整型索引：
# s = pd.Series([1,3,5,np.nan,6,8])
# print(type(s))

# concat
# 相同列合并到一起，不同时新建列，缺省值为NaN

# df1 =pd.DataFrame({'a':[1,2,3],'b':[4,5,6]})
# df2 =pd.DataFrame({'a':['a','b','c'],'c':['e','f','g']})

# df = pd.concat([df1,df2])

# print('df1:\n{df1} \ndf2:\n{df2}\ndf:\n{df}'.format(df1=df1,df2=df2,df=df))

# # reset_index
# df.reset_index(inplace=True)
# print(df)

# # drop
# df.drop('index', axis=1,inplace=True)
# print(df)

# # reindex_axis
# df =df.reindex_axis(df1.columns, axis=1)
# print(df)

# df = pd.DataFrame(np.arange(8).reshape(4,2),columns= ['a','b'])
# print(df)
# df['c'] = df['a'].map(lambda x: x+1)
# print(df)

# df1 =pd.DataFrame({'a':['female','male'],'b':['male','female']})
# print(df1)
# df1['a'] =df1['a'].map({'female':0, 'male':1})
# print(df1)

#  isin()
# 判断某一列元素是否属于列表里的元素，返回True False列表，如果为True，则对该行元素进行操作，False时不操作

# df = pd.DataFrame({'columns1':['a','b','c'],'columns2':['c','d','e']})
# print(df)
# print(df.columns1.isin(['a','b']))
# df.columns1[df.columns1.isin(['a','b'])]= 'cc'
# print (df)

#  采用均值/出现次数设置missing值
# 对于一列数字，要获取平均值，如下：
# df =pd.DataFrame(np.arange(8).reshape(4,2),columns=['a','b'])
# print(df)
# median =df.a.dropna().median()
# print ('median:',str(median))

# 对于一列非数字，例如字符，要找到出现频率最高的字符赋值给missing值
# df2 = pd.DataFrame({'a':['a','b','a'],'c':['e','f','g']})
# print(df2)
# freq_max =df2.a.dropna().mode().values
# print(freq_max)


#  属性数字化----枚举
# 某一属性，其值表示为字符，且范围较少时，可选择使用枚举进行数字化

# 用np.unique()生成唯一化后的元素，在用enumerate()生成对应元组，转化为列表后生成字典。再对字典进行map操作，即可实现数值化。

# df2 =pd.DataFrame({'aa':['a','b','c','a'],'dd':['d','e','f','e']})
# print(np.unique(df2.aa))
# print(enumerate(np.unique(df2.aa)))
# unique_value =list(enumerate(np.unique(df2.aa)))
# print(unique_value)

# dict = {key:value for value,key in unique_value}

# for i in dict.keys():
#     print(i,':',dict[i])


# df2.aa = df2.aa.map(lambda x:dict[x]).astype(int)
# print(df2)

# 哑变量
# 作用条件与枚举类似，属性值范围不大时，会为每一个值生成新的一列。结果需要concat

# df =pd.DataFrame({'column1':['aa','bb','cc'], 'column2':['dd','ee','ff']})
# print(df)
# dummy_df_column1 =pd.get_dummies(df.column1)
# print(dummy_df_column1)
# df =pd.concat([df,dummy_df_column1],axis=1)
# print(df)

# 每个属性值对应一列，所以属性值很多的情况下不适用，会生成较大的df。将哑变量变换后的新的属性列直接concat到原来的df中即可。

#  loc()
# loc()对应列选取多行，第一个元素取行，第二个元素对应列，默认情况下为所有列

# df = pd.DataFrame({'a':[1,2,3,4],'b':[5,6,7,8]})
# print (df.loc[(df.a.values> 2)]) #取出a列中值大于2的所有列的值，原df的值不变
# print(df.loc[(df.a.values> 2),'a']) #只作用于a列，输出a列
# df.loc[(df.a.values >2),'a'] = 2 #对其赋值，则改变df的值
# print(df)

 # bining面元组合
# 主要用于产生一个有序分类

df =pd.DataFrame(np.arange(16).reshape(8,2),columns=['aa','bb'])
print(df)
# 先用pd.qcut()将数据分为若干份，对应新的一列，元素为一个范围字符串，仍然需要量化
df['cc'] = pd.qcut(df.bb,2) #cc加入到原df中，不需要concat
print(df)

# 如果DataFrame对列的引用两种方式相同，为什么要有这种区别？

# 分类后元素只是string，还要进行数字化，可以采用enumerate，dummy，factorize。
dummy_df =pd.get_dummies(df.cc).rename(columns=lambda x:'dummy_' + str(x) )

df =pd.concat([df,dummy_df],axis=1).drop(['cc','dd'],axis=1)

print (df)