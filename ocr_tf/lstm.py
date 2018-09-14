import cv2

# Hog 检测器 用来计算HOG描述子；HOG描述子的维数，由图片大小，检测窗口大小，块大小，细胞单元中直方图bin个数决定

# 所有训练样本的特征向量组成的矩阵，行数等于所有样本的个数，列数等于HOG描述子维数
import tensorflow as tf

a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
c = tf.matmul(a, b)

cfig = tf.ConfigProto(log_device_placement=True)
tf.global_variables_initializer()

with tf.Session(config=cfig) as sess:
  print(sess.run(c))