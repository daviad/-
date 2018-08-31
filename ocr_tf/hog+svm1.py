import cv2
import os
import numpy as np

# 创建 路径 键值对
trainPath = '/Users/dxw/Downloads/tf_car_license_dataset/tf_car_license_dataset/train_images/training-set'
labelPathDic = {}
fileNames = os.listdir(trainPath)
for fileName in fileNames:
    # print(fileName)
    if fileName == "letters":
        pass
    elif fileName == ".DS_Store":
        pass
    elif fileName == "chinese-characters":
        pass
    else:
        dirlist = os.listdir(os.path.join(trainPath, fileName))
        labelPathDic[fileName] = list(map(lambda y : os.path.join(trainPath, fileName, y),
                                     (filter(lambda x: x != ".DS_Store" ,dirlist))))

# print(labelPathDic)


def builidHog(src):
    img = cv2.imread(src)
    # Hog
    # 1.设置一些参数
    winSize = (30, 60)
    blockSize = (10, 20)
    blockStride = (5, 10)
    cellSize = (5, 5)
    nbins = 9
    winStride = (30, 60)

    # 2.创建hog
    hog = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins)
    # 调用,hist是一个特征向量,维数可以算出来
    hist = hog.compute(img, winStride, padding=(0,0))
    # print(hist)
    return hist.T


# 创建 hog 键值对
hogDic = {}
for key,value in labelPathDic.items():
       hogDic[key] = np.array(list(map(builidHog,value)))


# print(hogDic)

# 创建TrainData
def buildTrainData():
    data = []
    label = []
    for key, value in hogDic.items():
        for v in value:
            data.append(v)
            label.append(key)
    return np.array(data), np.array(label)
#
# SVM
# 创建，设置参数
svm = cv2.ml.SVM_create()
svm.setType(cv2.ml.SVM_C_SVC)
svm.setKernel(cv2.ml.SVM_RBF)
svm.setC(1)
svm.setGamma(0.00055556)

# 训练
X, Y = buildTrainData()
trainDataSet = cv2.ml.TrainData_create(X, cv2.ml.ROW_SAMPLE, Y)
svm.train(trainDataSet)

# 保存
svm.save('svm.xml')
#
# # 调用
# svm1 = cv2.ml.SVM_load('svm.xml')
#
# # 预测
# p = svm.predict(des) # p[1][0][0]才是label

