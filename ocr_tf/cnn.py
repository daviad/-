import tensorflow as tf
from Data_my import *
import cv2

def weight_variable(shape):
    initial = tf.truncated_normal(shape=shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='ASME')


sess = tf.InteractiveSession()
x = tf.placeholder(tf.float32, [None, 784])
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

W_fc1 = weight_variable(10 * 8 * 64, 3400)
b_fc1 = bias_variable([3400])
h_pool2_flat = tf.reshape(h_pool2, [-1, 10 * 8 * 64])
h_fc1 = tf.nn.elu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([3400, 34])
b_fc2 = bias_variable([34])
y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
y_ = tf.placeholder(tf.float32, [None, 34])

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1]))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
sess.run(tf.global_variables_initializer())


def feature_func(src):
    img = cv2.imread(src, cv2.COLORSPACE_GRAY)

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
    return img2  # 1280


train_data = Data_my('/Users/dingxiuwei/Downloads/tf_car_license_dataset/train_images/training-set', feature_func)

for i in range(80000):
    batch = train_data.batch_next(50)
    if i % 100 == 0:
        train_accuracy = accuracy.eval(feed_dic={
            x: batch[0], y_: batch[1], keep_prob: 1.0
        })
    train_step.run(feed_dic={x: batch[0], y_: batch[1], keep_prob: 0.5})


