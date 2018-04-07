
# coding: utf-8

# In[ ]:


"""A very simple MNIST classifier.
See extensive documentation at
http://tensorflow.org/tutorials/mnist/beginners/index.md
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# get_ipython().run_line_magic('pdb', 'on')

# Import data
from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('data_dir', './data/', 'Directory for storing data') # 把数据放在/tmp/data文件夹中
mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)   # 读取数据集

# In[ ]:


# 建立抽象模型（网络）
x = tf.placeholder(tf.float32, [None, 784]) # 占位符
y = tf.placeholder(tf.float32, [None, 10])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
a = tf.nn.softmax(tf.matmul(x, W) + b)

# 定义损失函数和训练模型
"""
cross_entropy = -tf.reduce_sum(y*tf.log(a))
首先，用 tf.log 计算 a（预测值） 的每个元素的对数。接下来，我们把 y (正确的值) 的每一个元素和 tf.log(a) 的对应元素相乘。最后，用 tf.reduce_sum 计算张量的所有元素的总和。（注意，这里的交叉熵不仅仅用来衡量单一的一对预测和真实值，而是所有100幅图片的交叉熵的总和。对于100个数据点的预测表现比单一数据点的表现能更好地描述我们的模型的性能。
"""
## 按行来计算原因：有数据格式决定的。 这里的数据格式：一行代表一个数据（图片和对应的数字）
# reduction_indices=[1]  按行 求和  参照mnist数据格式来理解
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y * tf.log(a), reduction_indices=[1]))  # 损失函数为交叉熵
optimizer = tf.train.GradientDescentOptimizer(0.5) # 梯度下降法，学习速率为0.5
train = optimizer.minimize(cross_entropy) # 训练目标：最小化损失函数

# 测试训练好的模型 tf.argmax(a, 1)按行查找 参照常用函数argmax()理解
correct_prediction = tf.equal(tf.argmax(a, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Train
sess = tf.InteractiveSession()      # 建立交互式会话
tf.global_variables_initializer().run()
for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    # print(batch_xs)
    train.run({x: batch_xs, y: batch_ys})
print(sess.run(accuracy,feed_dict={x:mnist.test.images,y:mnist.test.labels}))

