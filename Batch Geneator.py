#encoding=utf-8
import numpy as np
import cv2 as cv
import os
import tensorflow as tf 

def getTrianList():
    root_dir = "/Users/zhuxiaoxiansheng/Desktop/doc/SICA_data/YaleB"
    with open('/Users/zhuxiaoxiansheng/Desktop'+"/Yaledata.txt","w") as f:
        for file in os.listdir(root_dir):
            if len(file) == 23:
                f.write(root_dir+'/'+file+" "+ file[11:13] +"\n")


def load_file(example_list_file):
   lines = np.genfromtxt(example_list_file,delimiter=" ",dtype=[('col1', 'S120'), ('col2', 'i8')])
   examples = []
   labels = []
   for example,label in lines:
       examples.append(example)
       labels.append(label)
   return np.asarray(examples),np.asarray(labels),len(lines)   

def trans2tfRecord(trainFile,output_dir):
    _examples,_labels,examples_num = load_file(trainFile)
    filename = output_dir + '.tfrecords'
    writer = tf.python_io.TFRecordWriter(filename)
    for i,[example,label] in enumerate(zip(_examples,_labels)):
        example = example.decode("UTF-8")
        image = cv.imread(example)
        image = cv.resize(image,(192,168)) 
        image_raw = image.tostring()
        example = tf.train.Example(features=tf.train.Features(feature={
                'image_raw':tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_raw])),
                'label':tf.train.Feature(int64_list=tf.train.Int64List(value=[label]))                        
                }))
        writer.write(example.SerializeToString())
    writer.close()
    return filename

def read_tfRecord(file_tfRecord):
    queue = tf.train.string_input_producer([file_tfRecord])
    reader = tf.TFRecordReader()
    _,serialized_example = reader.read(queue)
    features = tf.parse_single_example(
            serialized_example,
            features={
          'image_raw':tf.FixedLenFeature([], tf.string),   
          'label':tf.FixedLenFeature([], tf.int64)
                    }
            )
    image = tf.decode_raw(features['image_raw'],tf.uint8)
    image = tf.reshape(image,[192,168,3])
    image = tf.cast(image, tf.float32)
    image = tf.image.per_image_standardization(image)
    label = tf.cast(features['label'], tf.int64)
    return image,label



    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    


    
    
    
    
    
    
    
    
    
    
    
    
    
    
