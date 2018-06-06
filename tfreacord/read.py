#encoding=utf-8   
# 设置utf-8编码，方便在程序中加入中文注释．  
import os  
import scipy.misc  
import tensorflow as tf  
import numpy as np  
from test import *  
import matplotlib.pyplot as plt  
  
def read_and_decode(filename_queue):  
          
    reader = tf.TFRecordReader()  
    _, serialized_example = reader.read(filename_queue)  
      
    features = tf.parse_single_example(serialized_example,features = {  
                        'image_raw':tf.FixedLenFeature([], tf.string)})  
    image = tf.decode_raw(features['image_raw'], tf.uint8)  
    image = tf.reshape(image, [OUTPUT_SIZE, OUTPUT_SIZE, 3])  
    image = tf.cast(image, tf.float32)  
    #image = image / 255.0  
      
    return image  
  
data_dir = '/home/sanyuan/dataset_animal/dataset_tfrecords/'   
  
filenames = [os.path.join(data_dir,'train%d.tfrecords' % ii) for ii in range(1)]　#如果有多个文件，直接更改这里即可  
filename_queue = tf.train.string_input_producer(filenames)  
image = read_and_decode(filename_queue)  
with tf.Session() as sess:      
    coord = tf.train.Coordinator()  
    threads = tf.train.start_queue_runners(coord=coord)  
    for i in xrange(2):  
        img = sess.run([image])  
        print(img[0].shape)  # 设置batch_size等于１．每次读出来只有一张图  
        plt.imshow(img[0])  
        plt.show()  
    coord.request_stop()  
    coord.join(threads)  