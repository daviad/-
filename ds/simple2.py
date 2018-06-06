#!/usr/bin/python3
#coding:utf-8


'''
梯度下降法的原理和公式这里不讲，就是一个直观的、易于理解的简单例子。
1.最简单的情况，样本只有一个变量，即简单的（x，y）。多变量的则可为使用体重或身高判断男女（这是假设，并不严谨），则变量有两个，一个是体重，一个是身高，则可表示为(x1,x2,y),即一个目标值有两个属性。

2.单个变量的情况最简单的就是，函数hk(x)=k*x这条直线(注意：这里k也是变化的，我们的目的就是求一个最优的   k)。而深度学习中，我们是不知道函数的，也就是不知道上述的k。   这里讨论单变量的情况：

  在不知道k的情况下，我们是通过样本(x1,y1),(x2,y2)，(xn,yn)来获取k。获取的k的好坏则有损失函数来衡量。

  损失函数：就是你预测的值和真实值的差异大小（比如一个样本（1，1）他的真实值是1，而你预测的是0.5，则差异   比较大，如果你预测值为0.9999，则差异就比较小了）。

  损失函数为定义如下（此处为单变量的情况）

   

  目的是求使损失函数最小的变量k（注意和变量x区分），则将损失函数对k求导（多变量时为求偏导得梯度，这里单变量求导，其实不算梯度），求偏导如下：

   

  然后迭代，迭代时有个步长alpha,(深度学习中貌似叫学习率)


3.例子     

    假如我们得到样本(1,1),(2,2),(3,3).其实，由这三个样本可以得到函数为y = 1*x。此时损失函数为0.而机器是不知道的，所以我们需要训练。
    1.假设模型为y = kx 
    2.误差函数为
    3.梯度迭代 


    下面是一段python代码。    
'''

import numpy as np
import matplotlib.pyplot as plt

data = [[1,1],[2,2],[3,3]]

k = 3
k_old = k
k_new = 6
alpha = 0.01
precision = 0.0001
m = 1

count = 0

def hfun(x):
	return k*x

def derivative(x,y):
	return (hfun(x) - y)*x

# for i in xrange(0,1):

# 	tempData = data[i]
# 	x = tempData[0]
# 	y = tempData[1]
# 	print('x:'+str(x)+' ,	y:'+str(y))
# 	d = derivative(x,y)
# 	print d
# 	k = k - alpha * d
# 	print k



while abs(k_old - k_new) > precision:
# while  count > 0:
	count += 1
	for i in xrange(0,1):
		tempData = data[i]
		x = tempData[0]
		y = tempData[1]
		print('x:'+str(x)+' ,	y:'+str(y))
	d = derivative(x,y)
	print('derivative:'+str(d))
	k_old = k
	k = k - alpha * d
	k_new = k;
	print('k:'+str(k))

print('count:'+str(count))






