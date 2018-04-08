
# coding: utf-8
"""

下载下来的数据集被分成两部分：60000行的训练数据集（mnist.train）和10000行的测试数据集（mnist.test）。这样的切分很重要，在机器学习模型设计时必须有一个单独的测试数据集不用于训练而是用来评估这个模型的性能，从而更加容易把设计的模型推广到其他数据集上（泛化）。

正如前面提到的一样，每一个MNIST数据单元有两部分组成：一张包含手写数字的图片和一个对应的标签。我们把这些图片设为“xs”，把这些标签设为
“ys”。训练数 据集和测试数据集都包含xs和ys，比如训练数据集的图片是 mnist.train.images ，训练数据集的标签是mnist.train.labels。 

每一张图片包含28X28个像素点。我们可以用一个数字数组来表示这张图片：


我们把这个数组展开成一个向量，长度是 28x28=784。如何展开这个数组（数字间的顺序）不重要，只要保持各个图片采用相同的方式展开。从这个角度来看，MNIST数据集的图片就是在784维向量空间里面的点, 并且拥有比较复杂的结构 (提醒: 此类数据的可视化是计算密集型的)。

展平图片的数字数组会丢失图片的二维结构信息。这显然是不理想的，最优秀的计算机视觉方法会挖掘并利用这些结构信息，我们会在后续教程中介绍。但是在这个教程中我们忽略这些结构，所介绍的简单数学模型，softmax回归(softmax regression)，不会利用这些结构信息。

因此，在MNIST训练数据集中，mnist.train.images 是一个形状为 [60000, 784] 的张量，第一个维度数字用来索引图片，第二个维度数字用来索引每张图片中的像素点。在此张量里的每一个元素，都表示某张图片里的某个像素的强度值，值介于0和1之间。


相对应的MNIST数据集的标签是介于0到9的数字，用来描述给定图片里表示的数字。为了用于这个教程，我们使标签数据是"one-hot vectors"。 一个one-hot向量除了某一位的数字是1以外其余各维度数字都是0。所以在此教程中，数字n将表示成一个只有在第n维度（从0开始）数字为1的10维向量。比如，标签0将表示成([1,0,0,0,0,0,0,0,0,0,0])。因此， mnist.train.labels 是一个 [60000, 10] 的数字矩阵。

"""


# Import data
from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf
import numpy as np

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('data_dir', './data/', 'Directory for storing data') # 第一次启动会下载文本资料，放在./data文件夹下

print(FLAGS.data_dir)
mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)
print(mnist)
# mnist是一个轻量级的类。它以Numpy数组的形式存储着训练、校验和测试数据集。同时提供了一个函数(next_batch())，用于在迭代中获得minibatch
batchCount = 1 #这里对应行数

print(mnist.train.next_batch(batchCount))   
print("batch's type: {type}".format(type=type(mnist.train.next_batch(batchCount))))
#tuple代表一个图片和对应的label。
images_x,label_y = mnist.train.next_batch(batchCount);
print("image shape:{image_shape}".format(image_shape=np.shape(images_x)));
print("label shape:{label_shape}".format(label_shape=np.shape(label_y)));

"""
image 一行 28*28=784 个像素 代表一个数字的图片
label -行  1*10 （one-hot） 代表一个数字
"""

"""
print type(someObject).__name__
如果这不适合你，就用这个：
print some_instance.__class__.__name__

"".join(list(s))
'xxxxx'
>>> str(tuple(s))
"""
 

print('mnist.train.images type:'+ type(mnist.train.images).__name__ + "," +'shape'+str(np.shape(mnist.train.images)))
print('mnist.train.labels type:'+ type(mnist.train.labels).__name__ + "," +'shape'+str(np.shape(mnist.train.labels))) 





