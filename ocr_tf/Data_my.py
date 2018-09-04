import numpy as np
import os
import random


class Data_my:
    def __init__(self, path, feature_func):
        self.path = path
        self.sample = None
        self.batch_index = 0
        self.feature_func = feature_func

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
        data = []
        label = []
        for key, values in label_path_dic.items():
            for v in values:
                data.append(self.feature_func(v))
                label.append(key)
        # return data, label
        tmp = list(zip(data, label))
        self.sample = np.array(tmp)
        np.random.shuffle(self.sample)
        feature = self.sample[:, 0].tolist()
        label = self.sample[:, 1].tolist()
        return feature, label

    def batch_next(self, count):
        start = self.batch_index
        self.batch_index = self.batch_index + count
        if self.batch_index > self.sample.size:
            self.batch_index = self.sample.size

        end = self.batch_index
        if end == start:
            start = 0
            end = count
            self.batch_index = count

        return self.sample[start:end]


a = [[1, 12, 3], [2, 12, 3], [3, 12, 3]]
b = [1, 2, 1, 2]
c = zip(a, b)
e = list(c)
print(e)
random.shuffle(e)
print(e)
# f = zip(*c))
# print(f)
# f = np.array(e)
# print(f)
# print(f.shape)
# print(np.array(f[0]))
# # print(c[])
# print(e[1:3])
# print(e[1:3][0])
