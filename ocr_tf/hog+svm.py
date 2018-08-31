import cv2

img = cv2.imread("t.png")
# Hog
# 1.设置一些参数
winSize = (30,60)
blockSize = (10,20)
blockStride = (5,10)
cellSize = (5,5)
nbins = 9
winStride = (30,60)

# 2.创建hog
hog = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins)

# 调用,hist是一个特征向量,维数可以算出来
hist = hog.compute(img, winStride, padding=(0,0))
i = hist.reshape(-1,300)
print(i.shape)
print(type(hist))
print(hist.T.shape)
cv2.imshow("hog",i)
cv2.waitKey(0)
cv2.destroyAllWindows()

# # SVM
# # 创建，设置参数
# svm = cv2.ml.SVM_create()
# svm.setType(cv2.ml.SVM_C_SVC)
# svm.setKernel(cv2.ml.SVM_RBF)
# svm.setC(1)
# svm.setGamma(0.00055556)
#
# # 训练
# trainDataSet = cv2.ml.TrainData_create(X, cv2.ml.ROW_SAMPLE, Y)
# svm.train(trainDataSet)
#
# # 保存
# svm.save('svm.xml')
#
# # 调用
# svm1 = cv2.ml.SVM_load('svm.xml')
#
# # 预测
# p = svm.predict(des) # p[1][0][0]才是label

