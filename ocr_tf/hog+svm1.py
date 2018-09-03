import cv2
import os
import numpy as np
import time

class HogSvm(object):
    def __init__(self, train_path, test_path):
        self.train_path = train_path
        self.test_path = test_path
        self.label_path_dic = {}
        self.label_hog_dic = {}
        self.svm = None

    # 创建 路径 键值对
    def build_path_label_dic(self, path):
        for file_name in os.listdir(path):
            if file_name == "letters":
                pass
            elif file_name == ".DS_Store":
                pass
            elif file_name == "chinese-characters":
                pass
            else:
                self.label_path_dic[file_name] = list(map(lambda y: os.path.join(path, file_name, y),
                                                          (filter(lambda x: x != '.DS_Store',
                                                                  os.listdir(os.path.join(path, file_name))))))

    def build_hog(self, src):
        img = cv2.imread(src)
        # Hog
        # 1.设置一些参数
        win_size = (16, 20)
        win_stride = (16, 20)
        block_size = (8, 10)
        block_stride = (4, 5)
        cell_size = (4, 5)
        n_bins = 9

        # 2.创建hog
        hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, n_bins)
        hist = hog.compute(img, winStride=win_stride, padding=(0, 0))
        return hist.T.tolist()[0]  # 当矩阵是1 * n维的时候，经常会有tolist()[0]

        # img2 = np.reshape(img, [1, -1]).tolist()[0]
        # return img2


    # 创建 hog 键值对
    def build_hog_label_dic(self):
        for key, values in self.label_path_dic.items():
            self.label_hog_dic[key] = map(self.build_hog, values)  # np.array(list(map(builidHog,value)))

    # 创建TrainData
    def build_train_data(self):
        data = []
        label = []
        for key, value in self.label_hog_dic.items():
            for v in value:
                data.append(v)
                label.append(key)
        return data, label

    # SVM
    # 创建，设置参数
    def config_svm(self):
        self.svm = cv2.ml.SVM_create()
        self.svm.setType(cv2.ml.SVM_C_SVC)
        self.svm.setKernel(cv2.ml.SVM_LINEAR)
        self.svm.setC(1)
        self.svm.setGamma(0.00055556)

    # 训练
    def train(self):
        print('create train data...')
        self.build_path_label_dic(self.train_path)
        self.build_hog_label_dic()
        x, y = self.build_train_data()
        x = np.array(x, dtype=np.float32)
        y = np.array(y, dtype=np.int32)
        print('train svm...')
        start = time.clock()
        self.config_svm()
        train_data_set = cv2.ml.TrainData_create(x, cv2.ml.ROW_SAMPLE, y)
        self.svm.train(train_data_set)
        # 保存
        self.svm.save('svm.xml')
        end = time.clock()
        print('Running time: %.2f min' % ((end - start)/60))

    def predict(self, src):
        _svm = cv2.ml.SVM_load('svm.xml')
        des = self.build_hog(src)
        des = np.array(des, dtype=np.float32)
        p = _svm.predict(des.reshape(1, -1))
        print(p[1][0][0])

    def test(self):
        print('create test data...')
        self.build_path_label_dic(self.test_path)
        self.build_hog_label_dic()
        x, y = self.build_train_data()
        x = np.array(x, dtype=np.float32)
        y = np.array(y, dtype=np.int32)
        print('test svm...')
        start = time.clock()
        self.svm = cv2.ml.SVM_load('svm.xml')
        # 预测
        p = self.svm.predict(x)
        result = p[1]
        count = 0
        for i in range(0, result.size):
            if int(result[i][0]) == y[i]:
                count = count + 1

        precise = float(count)/float(result.size)
        print('precise:', precise)
        end = time.clock()
        print('Running time: %.2f min' % ((end - start)/60.0))
        # p[1][0][0]才是label


hogSVM = HogSvm('/Users/dingxiuwei/Downloads/tf_car_license_dataset/train_images/training-set/', '/Users/dingxiuwei/Downloads/tf_car_license_dataset/train_images/validation-set/')
hogSVM.train()
hogSVM.predict('/Users/dingxiuwei/Downloads/tf_car_license_dataset/train_images/training-set/26/1509808379_364_3_new_warped3.bmp')
hogSVM.test()

