import numpy as np
import os
import random

import tensorflow as tf


class Data_my:
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

    def convert2array(self, lst):
        return np.array(lst)

    def build_feature_label_nparray(self):
        label_path_dic = self.build_path_label_dic()
        # data = []
        # label = []
        self.category = len(label_path_dic)
        sum_tmp = []
        row = 0
        for key, values in label_path_dic.items():
            for v in values:
                row = row + 1
                # data.append((self.feature_func(v)))
                # label.append(key)
                sum_tmp.extend(self.feature_func(v))
                sum_tmp.append(key)
        # return data, label
        tmp = np.array(sum_tmp)
        self.sample = tmp.reshape(row, -1)
        np.random.shuffle(self.sample)
        # tmp = list(zip(data, label))
        # random.shuffle(tmp)
        # self.sample = tmp
        # tmp_arr = np.array(tmp)
        # feature = tmp_arr[:, 0].tolist()
        # label = tmp_arr[:, 1].tolist()
        # return feature, label

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


# a = [[1, 12, 3], [2, 12, 3], [3, 12, 3]]
# b = [1, 2, 1, 2]
# c = zip(a, b)
# e = list(c)
# print(e)
# random.shuffle(e)
# print(e)
# f = zip(*c))
# print(f)
# f = np.array(e)
# print(f)
# print(f.shape)
# print(np.array(f[0]))
# # print(c[])
# print(e[1:3])
# print(e[1:3][0])


# # one-hot
# a = [1,2,3,5,7]
# one_hot = tf.one_hot(a,7)
#
# tf.global_variables_initializer()
# with tf.Session() as sess:
#     sess.run(one_hot)
#     print(one_hot)

# a0 = [1,1,1]
# a1 = [2,2,2]
# a2 = [1,1,1]
#
# label = [0,1,0]
#
# d = np.array([])
#
# c = np.append(a0,label[0])
# c = np.append(c,a1)
# c = np.append(c,label[1])
# c = np.append(c,a2)
# c = np.append(c,label[2])
# print(c)
# c = []
# c.append(a0)
# c.append(a1)
# c.append(a2)

# tmp = list(zip(c, label))
# random.shuffle(tmp)
# d = np.array(tmp)
# print(d)

# x = [[0]*2]
# input_images = np.array([[0] * 2 for i in range(3)])
# print(input_images)