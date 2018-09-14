from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import cv2
import numpy as np
import os
import random
from Data_my import *
# https://blog.csdn.net/aliceyangxi1987/article/details/71716596
# CNN
def weight_variable(shape):
    initial = tf.truncated_normal(shape)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')


def conv_layer(x, shape, elu=True, pool=True):
    w = weight_variable(shape)
    b = []
    b.append(shape[-1])
    b = bias_variable(b)
    conv2d = tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME') + b
    h = conv2d
    if elu:
        h = tf.nn.elu(conv2d)
    if pool:
        h = max_pool_2x2(h)
    else:
        h = tf.nn.max_pool(h, ksize=[1, 1, 1, 1],
                       strides=[1, 1, 1, 1], padding='SAME')
    return h


def fc_layer1(x, shape, elu=True):
    w = weight_variable(shape)
    b = bias_variable(shape[-1:])
    flat = tf.reshape(x, [-1, shape[0]])
    h = tf.matmul(flat, w) + b
    if elu:
        h = tf.nn.elu(h)
    return h


def fc_layer2(x, shape):
    w = weight_variable(shape)
    b = bias_variable(shape[-1:])
    y = tf.matmul(x, w) + b
    y = tf.nn.softmax(y)
    return y


# cnn  网络变量
img_width = 28
img_height = 28
num_class = 10
lr = 1e-4
num_iter = 1000
batch_size = 60
display_iter = 80

x = tf.placeholder(tf.float32, [None, img_width * img_height])
keep_prob = tf.placeholder(tf.float32)
y_ = tf.placeholder(tf.float32, [None, num_class])

x_image = tf.reshape(x, [-1, img_height, img_width, 1])
# cnn 构建网络结构
h = conv_layer(x_image, [5, 5, 1, 32])
h_shape = h.shape.as_list()
h = conv_layer(h, [5, 5, h_shape[-1], 64], pool=True)
h_shape = h.shape.as_list()
h = fc_layer1(h, [h_shape[1] * h_shape[2] * h_shape[3], 1024])
h = tf.nn.dropout(h, keep_prob)
y = fc_layer2(h, [1024, num_class])
# cnn 定义网络优化
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_, logits=y))
train_step = tf.train.AdamOptimizer(lr).minimize(loss)
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
mnist = input_data.read_data_sets("./data", one_hot=True)

for i in range(num_iter):
    batch = mnist.train.next_batch(batch_size)
    train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob:0.5})
    if i % display_iter == 0:
        train_accuracy = accuracy.eval(feed_dict={x: batch[0], y_: batch[1], keep_prob:1})
        print("step:%d  train accuracy:%g"%(i, train_accuracy))

saver = tf.train.Saver()

saver.save(sess, "./cnn_model/")