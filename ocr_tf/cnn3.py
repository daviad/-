import tensorflow as tf
import cv2
import numpy as np
import os
import random
from Data_my import *

# 数据预处理
class DataMy:
    def __init__(self, path, feature_func):
        self.path = path
        self.sample = None
        self.batch_index = 0
        self.feature_func = feature_func
        self.category = 0

    # 创建 路径 键值对
    def build_path_label_dic(self):
        label_path_dic = {}
        for file_name in os.listdir(self.path):
            if file_name == "letters":
                pass
            elif file_name == ".DS_Store":
                pass
            elif file_name == "chinese-characters":
                pass
            else:
                label_path_dic[file_name] = list(map(lambda y: os.path.join(self.path, file_name, y),
                                                     (filter(lambda x: x != '.DS_Store',
                                                             os.listdir(os.path.join(self.path, file_name))))))
        return label_path_dic

    def build_feature_label_nparray(self):
        label_path_dic = self.build_path_label_dic()
        self.category = len(label_path_dic)
        sum_tmp = []
        row = 0
        for key, values in label_path_dic.items():
            for v in values:
                row = row + 1

                sum_tmp.extend(self.feature_func(v))
                sum_tmp.append(key)
        tmp = np.array(sum_tmp)
        self.sample = tmp.reshape(row, -1)
        np.random.shuffle(self.sample)

    def batch_next(self, count):
        row = self.sample.shape[0]
        col = self.sample.shape[1]
        start = self.batch_index
        self.batch_index = self.batch_index + count
        if self.batch_index > row:
            self.batch_index = row

        end = self.batch_index
        if end == start:
            start = 0
            end = count
            self.batch_index = count

        tmp = self.sample[start:end, :]
        feature = np.array(tmp[:, 0:(col -1)], dtype=np.float32)
        label = tmp[:, (col - 1):col]
        one_hot = tf.one_hot(label, self.category, dtype=tf.float32)
        one_hot = one_hot.eval().flatten().reshape(-1, self.category)
        t = (feature, one_hot)
        return t


def feature_func(src):
    img = cv2.imread(src, 0)
    _, img2 = cv2.threshold(img, 230, 1, cv2.THRESH_BINARY)
    img3 = np.reshape(img2, [1, -1]).tolist()[0]
    return img3  # 1280


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


# train_data = Data_my('/Users/dxw/Downloads/tf_car_license_dataset/tf_car_license_dataset/train_images/training-set/chinese-characters', feature_func)
# train_data.build_feature_label_nparray()

# cnn  网络变量
img_width = 32
img_height = 40
num_class = 34  # train_data.category
lr = 1e-4
num_iter = 1000
batch_size = 60
display_iter = 80

x = tf.placeholder(tf.float32, [None, img_width * img_height])
keep_prob = tf.placeholder(tf.float32)
y_ = tf.placeholder(tf.float32, [None, num_class])

x_image = tf.reshape(x, [-1, img_height, img_width, 1])
# cnn 构建网络结构
h = conv_layer(x_image, [8, 8, 1, 16])
h = conv_layer(h, [5, 5, 16, 32], pool=False)
h_shape = h.shape.as_list()
h = fc_layer1(h, [h_shape[1] * h_shape[2] * h_shape[3], 512])
h = tf.nn.dropout(h, keep_prob)
y = fc_layer2(h, [512, num_class])
# cnn 定义网络优化
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_, logits=y))
train_step = tf.train.AdamOptimizer(lr).minimize(loss)
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

# for i in range(num_iter):
#     batch = train_data.batch_next(batch_size)
#     train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob:0.5})
#     if i % display_iter == 0:
#         train_accuracy = accuracy.eval(feed_dict={x: batch[0], y_: batch[1], keep_prob:1})
#         print("step:%d  train accuracy:%g"%(i, train_accuracy))

saver = tf.train.Saver()

saver.save(sess, "./cnn_model/")