# 十分钟搞定pandas https://www.cnblogs.com/chaosimple/p/4153083.html

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 创建对象 
# 1、可以通过传递一个list对象来创建一个Series，pandas会默认创建整型索引：
s = pd.Series([1,3,5,np.nan,6,8])
print(s)