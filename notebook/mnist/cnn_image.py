#!python3
'''
参考：TensorFlow入门（三）多层 CNNs 实现 mnist分类
https://blog.csdn.net/jerr__y/article/details/57086434

'''

#coding: utf-8

# Import data
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

mnist = input_data.read_data_sets('./data', one_hot=True)   # 读取数据集




# 我们先来看看数据是什么样的
img1 = mnist.train.images[1]
label1 = mnist.train.labels[1]
print(label1)  # 所以这个是数字 6 的图片
print("img_data shape = {shape}".format(shape=img1.shape))  # 我们需要把它转为 28 * 28 的矩阵
print("img_data shape",img1.shape)
img1.shape = [28, 28]

import matplotlib.pyplot as plt
# import matplotlib.image as mpimg  # 用于读取图片，这里用不上
# print(img1.shape)
# plt.imshow(img1)
# # plt.axis('off') # 不显示坐标轴
# plt.show()  

# # 我们可以通过设置 cmap 参数来显示灰度图
# plt.imshow(img1, cmap='gray') # 'hot' 是热度图
# plt.show()

##  具体的实现在cnn.py zhong

# 首先应该把 img1 转为正确的shape (None, 784)

