import tensorflow as tf
import numpy as np
from Data_my import *
from tensorflow.contrib import rnn
import cv2

# https://blog.csdn.net/jmh1996/article/details/78821216

train_rate = 0.001
train_iter = 10000
batch_size = 128
frame_size = 32
sequence_length = 40
hidden_num = 5
n_class = 34

display_step = 100

x = tf.placeholder(tf.float32, [None, batch_size*sequence_length], name="inputx")
y = tf.placeholder(tf.float32, [None, n_class], name="expected_y")
weights = tf.Variable(tf.truncated_normal([hidden_num, n_class]))
bias = tf.Variable(tf.zeros([n_class]))


def RNN(x, weights, biase):
    x = tf.reshape(x, [-1, sequence_length, frame_size])
    rnn_cell = rnn.LSTMCell(hidden_num)
    output, states = tf.nn.dynamic_rnn(rnn_cell, x, dtype=tf.float32)
    return tf.nn.softmax(tf.matmul(output[:,-1,:],weights)+bias,1)


predy=RNN(x,weights,bias)
cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=predy,labels=y))
train=tf.train.AdamOptimizer(train_rate).minimize(cost)

correct_pred=tf.equal(tf.argmax(predy,1),tf.argmax(y,1))
accuracy=tf.reduce_mean(tf.to_float(correct_pred))

sess=tf.Session()
sess.run(tf.initialize_all_variables())
step=1

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
    x ,img2 = cv2.threshold(img, 230, 1, cv2.THRESH_BINARY)
    img3 = np.reshape(img2, [1, -1]).tolist()[0]
    # img2 = np.reshape(img, [1, -1])
    return img3  # 1280

train_data = Data_my('/Users/dxw/Downloads/tf_car_license_dataset/tf_car_license_dataset/train_images/training-set/chinese-characters', feature_func)
train_data.build_feature_label_nparray()

# testx,testy = train_data.next_batch(batch_size)
while step<train_iter:
    batch_x,batch_y= train_data.batch_next(batch_size)
#    batch_x=tf.reshape(batch_x,shape=[batch_size,sequence_length,frame_size])
    _loss,__=sess.run([cost,train],feed_dict={x:batch_x,y:batch_y})
    if step % display_step ==0:

        acc,loss=sess.run([accuracy,cost],feed_dict={x:batch_x,y:batch_y})
        print(step,acc,loss)

    step += 1