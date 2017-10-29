#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  6 16:01:24 2017

@author: llc
"""
#%% " 用cv2.imwrite 与cv2.imencod两种方式生成"
import os
import glob
import cv2
import numpy as np
import tensorflow as tf

from genplate import GenPlate
from input_data import gen_sample

G = GenPlate("./font/platech.ttf",'./font/platechar.ttf',"./NoPlates")
#G.genBatch(15,2,range(31,65),"./plate",(272,72))


plate_dir = '/home/llc/TF_test/Chinese_plate_recognition/Plate_recognition/plate/'


def gen_txt_plate(num,width, height):
    with open(os.path.join(plate_dir, 'test.txt'), 'w') as f_txt:
        for i in range(num):
            image_list=[]
            label ,img = gen_sample(G,width, height)
            img = np.multiply(img,255.0) #dtype = float64
            img = img.astype('uint8')  #dtype = uint8
            print(img.shape)
            #cv2.imwrite(plate_dir + str(i).zfill(2) + ".jpg", img)
            cv2.imencode('.jpg',img, [cv2.IMWRITE_JPEG_QUALITY,95])[1].tofile(plate_dir+str(i).zfill(2)+'.jpg')
            #image_list = os.listdir(plate_dir)
           # image_list =glob.glob(os.path.join(plate_dir,str(i).zfill(2),'.jpg'))
            img_list =[plate_dir,str(i).zfill(2),'.jpg']
            image_list =''.join(img_list)
           # image_list.append(' '+str(label))
            
           # f_txt.write(image_list + ' '+str(label))
            f_txt.write(image_list + ' '+repr(label)) # 重点http://blog.csdn.net/jmilk/article/details/49720611
            f_txt.write('\n')
        
if __name__=='__main__':
    gen_txt_plate(2,272,72)


#%%  测试生成list型标签的txt文件
''' Reference: http://blog.csdn.net/helei001/article/details/51354404'''

import os
import re
plate_dir = '/home/llc/TF_test/Chinese_plate_recognition/Plate_recognition/plate/'

#with open(os.path.join(plate_dir, 'test.txt'),'r') as f_test:
#    image_list,label_list=[re.split(' |\n',l,1)[0][1] for l in f_test.readlines()]
   # label_list=[re.split(' |\n',l,1)[1] for l in f_test.readlines()]
   
    #image_list=[re.split(' |\n',l,1)[0] for l in f_test.readlines()]
    #label_list=[l.split(' ',1)[1] for l in f_test.readlines()]
    #image_list=[str(l.split(' ',1)[0]) for l in f_test.readlines()]
 
image_list = [str(l.split(' ',1)[0]) for l in open(os.path.join(plate_dir, 'test.txt'),'r')]
label_list = [eval(l.split(' ',1)[-1]) for l in open(os.path.join(plate_dir, 'test.txt'),'r')]
## 用eval原因 http://blog.csdn.net/jmilk/article/details/49720611
print(image_list)
print(label_list)
#%%  "用tf.image.encode_jpeg生成"

import os
import cv2
import numpy as np
import tensorflow as tf

from genplate import GenPlate
from input_data import gen_sample

G = GenPlate("./font/platech.ttf",'./font/platechar.ttf',"./NoPlates")
#G.genBatch(15,2,range(31,65),"./plate",(272,72))


plate_dir = '/home/llc/TF_test/Chinese_plate_recognition/Plate_recognition/plate/'

def tf_encode(img,filepath):
    g0 = tf.Graph()
    with g0.as_default():
        write_pir = tf.cast(filepath,tf.string)
        data_image = tf.placeholder(tf.uint8)
        encode_img = tf.image.encode_jpeg(data_image,format='rgb')
        write_file =tf.write_file(write_pir,encode_img)
        init = tf.global_variables_initializer()
    with tf.Session(graph=g0) as sess:
        sess.run(init)
        write_image = sess.run(write_file,feed_dict={data_image:img})
   
    return write_image
    

def gen_txt_plate(num,width, height):
    with open(os.path.join(plate_dir, 'test.txt'), 'w') as f_txt:
        for i in range(num):
            image_list = []
            filepath = plate_dir + str(i).zfill(2) + ".jpg"
            label ,img = gen_sample(G,width, height)
            img = np.multiply(img,255.0)
            
            tf_encode(img,filepath)
            
            img_list =[plate_dir,str(i).zfill(2),'.jpg']
            image_list =''.join(img_list)
            f_txt.write(image_list + ' '+repr(label)) # 重点http://blog.csdn.net/jmilk/article/details/49720611
            f_txt.write('\n')
        
if __name__=='__main__':
    gen_txt_plate(160,272,72)

###   