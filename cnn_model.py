#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 13 21:25:52 2017

@author: llc
"""
import sys
sys.path.insert(0,"../../python")
import tensorflow as tf
import numpy as np
import cv2, random
from io import BytesIO
from genplate import *


class OCRBatch(object):
    def __init__(self, data_names, data, label_names, label):
        self.data = data
        self.label = label
        self.data_names = data_names
        self.label_names = label_names
        
    @property
    def provide_data(self):
        return[(n,x.shape) for n,x in zip(self.data_names, self.data)]
    
    @property
    def provide_label(self):
        return[(n,x.shape) for n,x in zip(self.label_names,self.label)]

def rand_range(lo,hi):
    return lo+r(hi-lo)

def gen_rand():
    name = ""
    label=[]
    label.append(rand_range(0,31))  #产生车牌开头32个省的标签
    label.append(rand_range(41,65)) #产生车牌第二个字母的标签
    for i in range(5):
        label.append(rand_range(31,65)) #产生车牌后续5个字母的标签
        
    name+=chars[label[0]]
    name+=chars[label[1]]
    for i in range(5):
        name+=chars[label[i+2]]
    return name,label

def gen_sample(genplate, width, height):
    num,label =gen_rand()
    img = genplate.generate(num)
    img = cv2.resize(img,(width,height))
    img = np.multiply(img,1/255.0)
    img = img.transpose(2,0,1)
    return label,img

class OCRIter():
    def __init__(self, count, batch_size, num_label, height, width):
        super(OCRIter, self).__init__()
        self.genplate = GenPlate("./font/platech.ttf",'./font/platechar.ttf','./NoPlates')
        self.batch_size = batch_size
        self.count = count
        self.height = height
        self.width = width
        self.provide_data = [('data', (batch_size, 3, height, width))]
        self.provide_label = [('softmax_label', (self.batch_size, num_label))]
        print ("start")
    def __iter__(self):

        for k in range(self.count / self.batch_size):
            data = []
            label = []
            for i in range(self.batch_size):
                num, img = gen_sample(self.genplate, self.width, self.height)
                data.append(img)
                label.append(num)
            
            data_all = [data]
            label_all = [label]
            data_names = ['data']
            label_names = ['softmax_label']
            data_batch = OCRBatch(data_names, data_all, label_names, label_all)
            yield data_batch

    def reset(self):
        pass


        
def Accuracy(label, pred):
    label = label.T.reshape((-1, ))
    hit = 0
    total = 0
    for i in range(pred.shape[0] / 7):
        ok = True
        for j in range(7):
            k = i * 7 + j
            if np.argmax(pred[k]) != int(label[k]):
                ok = False
                break
        if ok:
            hit += 1
        total += 1
    return 1.0 * hit / total

def train():
    with tf.name_scope('optimizer'):
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

if __name__=='__main__':
    train()
        
        