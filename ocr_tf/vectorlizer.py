# -*- coding: utf-8 -*-
"""
实现函数的向量化
"""

from math import exp
import numpy as np


# 自定义函数sigmoid function
def tom_sigmoid(x):
    return 1.0 / (1 + exp(-1 * x))


# 方法1：for element in x
def tom_for():
    x = [1, 2, 3]
    ret = []
    for element in x:
        ret.append(tom_sigmoid(element))
    print(ret)
    # [0.7310585786300049, 0.8807970779778823, 0.9525741268224334]


# 方法2：[f(x) for x in X]
def tom_np_array():
    x = [1, 2, 3]
    ret = [tom_sigmoid(e) for e in x]
    print(ret)
    # [ 0.73105858  0.88079708  0.95257413]


# 方法3：function + np.frompyfunc()
def tom_np_frompyfunc():
    x = [1, 2, 3]  # <class 'numpy.ndarray'>
    tom_sigmoid_func = np.frompyfunc(tom_sigmoid, 1, 1)
    ret = tom_sigmoid_func(x)  # <class 'numpy.ndarray'>
    print(ret)
    # [0.7310585786300049 0.8807970779778823 0.9525741268224334]


# 方法5：np.vectorize()
def tom_np_vectorize():
    x = [1, 2, 3]  # <class 'numpy.ndarray'>
    tom_sigmoid_vec = np.vectorize(tom_sigmoid)
    ret = tom_sigmoid_vec(x)
    print(ret)
    # [ 0.73105858  0.88079708  0.95257413]


if __name__ == '__main__':
    tom_for()
    tom_np_array()
    tom_np_frompyfunc()
    tom_np_vectorize()

