import tensorflow as tf
import os
import numpy

# 创建 路径 键值对
def build_path_label_dic(self, path):
    label_path_dic = {}
    for file_name in os.listdir(path):
        if file_name == "letters":
            pass
        elif file_name == ".DS_Store":
            pass
        elif file_name == "chinese-characters":
            pass
        else:
            label_path_dic[file_name] = list(map(lambda y: os.path.join(path, file_name, y),
                                                      (filter(lambda x: x != '.DS_Store',
                                                              os.listdir(os.path.join(path, file_name))))))
    return label_path_dic


def build_path_label_pair(self, path):
    label_path_dic = self.build_path_label_dic()
    data = []
    label = []
    for key, values in label_path_dic.items():
        for v in values:
            data.append(v)
            label.append(key)
    return data, label

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
x_image = tf.reshape(x, [-1, 28, 28, 1])

W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.elu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.elu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

W_fc1 = weight_variable(7 * 7 * 64, 1024)
b_fc1 = bias_variable([1024])
h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
h_fc1 = tf.nn.elu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])
y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
y_ = tf.placeholder(tf.float32, [None, 10])

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1]))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
sess.run(tf.global_variables_initializer())

def train_data_batch(count):
    train_data = np.array([0])


for i in range(1000):
    batch = mnist.train.next_batch(50)
    if i%100 == 0:
        train_accuracy = accuracy.eval(feed_dic={
            x:batch[0], y_:batch[1], keep_prob:1.0
        })
    train_step.run(feed_dic={x:batch[0], y_:batch[1], keep_prob:0.5})


