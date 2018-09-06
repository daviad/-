import tensorflow as tf
from Data_my import *
import cv2

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1) # 变量的初始值为截断正太分布
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    """
    tf.nn.conv2d功能：给定4维的input和filter，计算出一个2维的卷积结果
    前几个参数分别是input, filter, strides, padding, use_cudnn_on_gpu, ...
    input   的格式要求为一个张量，[batch, in_height, in_width, in_channels],批次数，图像高度，图像宽度，通道数
    filter  的格式为[filter_height, filter_width, in_channels, out_channels]，滤波器高度，宽度，输入通道数，输出通道数
    strides 一个长为4的list. 表示每次卷积以后在input中滑动的距离
    padding 有SAME和VALID两种选项，表示是否要保留不完全卷积的部分。如果是SAME，则保留
    use_cudnn_on_gpu 是否使用cudnn加速。默认是True
    """
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    """
    tf.nn.max_pool 进行最大值池化操作,而avg_pool 则进行平均值池化操作
    几个参数分别是：value, ksize, strides, padding,
    value:  一个4D张量，格式为[batch, height, width, channels]，与conv2d中input格式一样
    ksize:  长为4的list,表示池化窗口的尺寸
    strides: 窗口的滑动值，与conv2d中的一样
    padding: 与conv2d中用法一样。
    """
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')


def feature_func(src):
    img = cv2.imread(src, 0)
    #
    # # Hog
    # # 1.设置一些参数
    # win_size = (16, 20)
    # win_stride = (16, 20)
    # block_size = (8, 10)
    # block_stride = (4, 5)
    # cell_size = (4, 5)
    # n_bins = 9
    #
    # # 2.创建hog
    # hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, n_bins)
    # hist = hog.compute(img, winStride=win_stride, padding=(0, 0))
    # return hist.T.tolist()[0]  # 当矩阵是1 * n维的时候，经常会有tolist()[0]

    img2 = np.reshape(img, [1, -1]).tolist()[0]
    # img2 = np.reshape(img, [1, -1])
    return img2  # 1280

train_data = Data_my('/Users/dingxiuwei/Downloads/tf_car_license_dataset/train_images/training-set', feature_func)
train_data.build_feature_label_nparray()

sess = tf.InteractiveSession()
x = tf.placeholder(tf.float32, [None, 1280])
x_image = tf.reshape(x, [-1, 40, 32, 1])

# W_conv0 = weight_variable([5, 5, 1, 16])
# b_conv0 = bias_variable([16])
# h_conv0 = tf.nn.elu(conv2d(x_image, W_conv0) + b_conv0)
# h_pool0 = max_pool_2x2(h_conv0)

W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.elu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.elu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

W_fc1 = weight_variable([10 * 8 * 64, train_data.category * 10])
b_fc1 = bias_variable([train_data.category * 10])
h_pool2_flat = tf.reshape(h_pool2, [-1, 10 * 8 * 64])
h_fc1 = tf.nn.elu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([train_data.category * 10, train_data.category])
b_fc2 = bias_variable([train_data.category])
y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
y_ = tf.placeholder(tf.float32, [None, train_data.category])

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1]))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
sess.run(tf.global_variables_initializer())


for i in range(100):
    batch = train_data.batch_next(50)
    if i % 100 == 0:
        train_accuracy = accuracy.eval(feed_dict={
            x: batch[0], y_: batch[1], keep_prob: 1.0
        })
        print("step %d, training accuracy %g" % (i, train_accuracy))
    train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

# print("test accuracy %g"%accuracy.eval(feed_dict={
#     x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))


