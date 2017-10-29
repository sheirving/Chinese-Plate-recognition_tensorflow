#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 17 08:37:30 2017

@author: llc
"""
#%%
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import datetime as datetime
import time
import math 

import tensorflow as tf
import numpy as np
import cv2 
import os
import model

index = {"京": 0, "沪": 1, "津": 2, "渝": 3, "冀": 4, "晋": 5, "蒙": 6, "辽": 7, "吉": 8, "黑": 9, "苏": 10, "浙": 11, "皖": 12,
         "闽": 13, "赣": 14, "鲁": 15, "豫": 16, "鄂": 17, "湘": 18, "粤": 19, "桂": 20, "琼": 21, "川": 22, "贵": 23, "云": 24,
         "藏": 25, "陕": 26, "甘": 27, "青": 28, "宁": 29, "新": 30, "0": 31, "1": 32, "2": 33, "3": 34, "4": 35, "5": 36,
         "6": 37, "7": 38, "8": 39, "9": 40, "A": 41, "B": 42, "C": 43, "D": 44, "E": 45, "F": 46, "G": 47, "H": 48,
         "J": 49, "K": 50, "L": 51, "M": 52, "N": 53, "P": 54, "Q": 55, "R": 56, "S": 57, "T": 58, "U": 59, "V": 60,
         "W": 61, "X": 62, "Y": 63, "Z": 64};
         
chars = ["京", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙", "皖", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤", "桂",
             "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "A",
             "B", "C", "D", "E", "F", "G", "H", "J", "K", "L", "M", "N", "P", "Q", "R", "S", "T", "U", "V", "W", "X",
             "Y", "Z"
             ];

batch_size = 16
capacity = 8
num_examples= 160

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('eval_log_dir','/home/llc/TF_test/Chinese_plate_recognition/Plate_recognition/eval_log/',
                           """Directory where to write event logs.""")

tf.app.flags.DEFINE_string('eval_data_dir','/home/llc/TF_test/Chinese_plate_recognition/Plate_recognition/plate/',
                           """eval_data_dir'.""")

tf.app.flags.DEFINE_string('checkpoint_dir','/home/llc/TF_test/Chinese_plate_recognition/Plate_recognition/train_logs_50000/',
                           """Directory where to read model checkpoints.""")

#tf.app.flags.DEFINE_integer('num_examples',,"""Number of examples to run.""")

img_test_holder = tf.placeholder(tf.float32,[batch_size,72,272,3])
#keep_prob =tf.placeholder(tf.float32)

def get_input_list():
    image_list = [str(l.split(' ',1)[0]) for l in open(os.path.join(FLAGS.eval_data_dir, 'test.txt'),'r')]
    label_list = [eval(l.split(' ',1)[-1]) for l in open(os.path.join(FLAGS.eval_data_dir, 'test.txt'),'r')]
    ## 用eval原因 http://blog.csdn.net/jmilk/article/details/49720611
    return image_list,label_list

def get_input_batch(image_list,label_list):
    image_list =tf.cast(image_list,tf.string)
    label_list = tf.cast(label_list,tf.int32)
    input_queue = tf.train.slice_input_producer([image_list,label_list],shuffle=False,num_epochs=1)

    label = input_queue[1]
    image_contents = tf.read_file(input_queue[0])
    image = tf.image.decode_jpeg(image_contents,channels=3)
#    image = cv2.imread(input_queue[0])
#    image = np.multiply(image,1/255.0)
    image = tf.cast(image,tf.float32)
    image = tf.multiply(image,1/255)
    image_batch,label_batch = tf.train.batch([image,label],batch_size=batch_size,capacity=capacity,shapes=[[72,272,3],[7,]])
    
    #image_batch = tf.cast(image_batch, tf.float32)
    return image_batch,label_batch
    
    

def eval_once(saver,summary_writer,logits,top_k_op,summary_op):
    """ Run Eval once
    Args:
        saver: Saver
        summary_writer: Summary writer
        logits: models output 
        summary_op: Summary op
   """
    with tf.Session() as sess:
        init_op = tf.local_variables_initializer()
        sess.run(init_op)
        print('Reading checkpoint .....')
        ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
           # Restores from checkpoint
            saver.restore(sess, ckpt.model_checkpoint_path)
            global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
            print('Loading success, global_step is %s' % global_step)
        else:
            print('No checkpoint file found')
            return
        #start the queue runners
        coord = tf.train.Coordinator()
        #threads = tf.train.start_queue_runners(sess=sess,coord=coord)
        try:
            threads = []
            for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
                threads.extend(qr.create_threads(sess, coord=coord, daemon=True,
                                         start=True))
            num_iter = int(math.ceil(num_examples / batch_size))
            true_count = 0  # Counts the number of correct predictions.
            total_sample_count = num_iter * batch_size
            step = 0
         
            flase_plate_num =[]
            while step < num_iter and not coord.should_stop():
                prediction,acc = sess.run([logits,top_k_op])
                
                ###
                acc_reshape = np.reshape(acc,[batch_size,7])
                #计算每个batch中识别正确的车牌数
                acc_batch = list(map(np.array_equal,acc_reshape,np.ones([batch_size,7],dtype=int)))
                true_count += np.sum(acc_batch)
            
                #识别错误的车牌编号
                acc_batch_int = list(map(int,acc_batch))
                print(acc_batch_int)
                first_pos = 0
                
                for i in range(acc_batch_int.count(0)):
                    new_list = acc_batch_int[first_pos:]
                    next_pos = new_list.index(0)+1
                    flase_plate_num.append(step*batch_size+first_pos+new_list.index(0))
                    first_pos += next_pos
                    
                ###
                
                for b in range(batch_size):
                    line = ''
                    for i in range(prediction.shape[1]):
                        if i == 0:
                            result = np.argmax(prediction[b][i][0:31])
                        if i == 1:
                            result = np.argmax(prediction[b][i][41:65])+41
                        if i > 1:
                            result = np.argmax(prediction[b][i][31:65])+31
                        
                        line += chars[result]+" "
                   # print ('batch_num:%d predicted %d: %s' % (step,b,line)) 
                    print('Plate_num:%d.jpg,predicted:%s' % (step*batch_size+b,line))
                step += 1  
                
            # Compute precision.       
            precision = true_count / total_sample_count
            print('%s: precision = %.3f' % (datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), precision))
            
            # recognizing flase
            print('finding recognize flase plat num: ',flase_plate_num)
            
            # 此部分仿照cifar-10例子，每10秒验证一次，且生成log(tensorboard)
#            summary = tf.Summary()
#            summary.ParseFromString(sess.run(summary_op))
#            summary.value.add(tag='Precision', simple_value=precision)
#            summary_writer.add_summary(summary, global_step)
#            summary_writer.close()
        except Exception as e:  # pylint: disable=broad-except
            coord.request_stop(e)
            
        coord.request_stop()
        coord.join(threads)
   
def evaluate():
    """Eval plate for a number of steps"""
   # g = tf.get_default_graph()
    #with g.as_default() :
    
    with tf.Graph().as_default() as g:
        #get images and labels
        #Build a Graph that computes the logits predictions from the inference model 
        image_list,label_list = get_input_list()
        image_batch,label_batch= get_input_batch(image_list,label_list)
        
        logits1,logits2,logits3,logits4,logits5,logits6,logits7  = model.inference(image_batch,1) #output=[7,b,65]
        logits = tf.stack([logits1,logits2,logits3,logits4,logits5,logits6,logits7],axis=1) #output=[b,7,65]
        logits_label= tf.reshape(tf.nn.softmax(logits),[-1,65])  #output=[b*7,65]
        label_batch = tf.reshape(label_batch,[-1])   # input:[b,7] , output:[b*7,1]
       # logit = model.inference(image_batch,keep_prob) #output=[7,b,65]
       # logits = tf.reshape(tf.nn.softmax(logit),[-1,65])   #output=[7*b,65]
       # label_batch = tf.reshape(tf.transpose(label_batch),[-1]) # input:[b,7] , output:[7*b,1]
        top_k_op = tf.nn.in_top_k(logits_label,label_batch,1)
        
        summary_op = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter(FLAGS.eval_log_dir,g)
        
        saver = tf.train.Saver()
        
        eval_once(saver,summary_writer,logits,top_k_op,summary_op)

def main(argv=None):
    if tf.gfile.Exists(FLAGS.eval_log_dir):
        tf.gfile.DeleteRecursively(FLAGS.eval_log_dir)
    tf.gfile.MakeDirs(FLAGS.eval_log_dir)
    
    evaluate()

if __name__=='__main__':
    tf.app.run()
    
    
