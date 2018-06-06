import tensorflow as tf  
import numpy as np  
import os  
from PIL import Image  
def _int64_feature(value):  
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))  
  
def _bytes_feature(value):  
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))  
  
  
def img_to_tfrecord(data_path):  
    rows = 256  
    cols = 256  
    depth = 3  
    writer = tf.python_io.TFRecordWriter('test.tfrecords')  
    labelfile=open("random.txt")  
    lines=labelfile.readlines()  
    for line in lines:  
        #print line  
        img_name = line.split(" ")[0]#name  
        label = line.split(" ")[1]#label  
        img_path = data_path+img_name  
        img = Image.open(img_path)  
        img = img.resize((rows,cols))  
        #img_raw = img.tostring()      
        img_raw = img.tobytes()   
        example = tf.train.Example(features = tf.train.Features(feature = {  
                            'height': _int64_feature(rows),  
                           'weight': _int64_feature(cols),  
                            'depth': _int64_feature(depth),  
                        'image_raw': _bytes_feature(img_raw),  
                'label': _bytes_feature(label)}))  
                  
            writer.write(example.SerializeToString())      
    writer.close()   
  
  
  
if __name__ == '__main__':  
    current_dir = os.getcwd()      
    data_path = current_dir + '/data/'      
    #name = current_dir + '/data'  
    print('Convert start')     
    img_to_tfrecord(data_path)  
    print('done!')  