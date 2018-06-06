#!/usr/bin/env python3
#coding:utf8

'''
一个梯度下降算法的例子
gradient descent

函数是f(x)=x**2+2 python写法
导数是 f’(x)=2*x

'''
import numpy as np
import matplotlib.pyplot as plt

x = np.arange(-50,50,0.001)
y = x**2+2


old = 0
new = 6
setp = 0.01
precision = 0.00001

def derivative(x):
	return 2*x

while abs(new - old) > precision:
	old = new
	d = derivative(new)
	new = new - setp * d
	print ('x:'+ str(new) + ' 		 d:' + str(d))

# print (new)

plt.plot(x,y)
plt.show()