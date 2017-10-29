#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  4 10:34:40 2017

@author: llc
"""
#%% test1
import sys
sys.path.insert(0, "../../python")
import numpy as np
import tensorflow as tf
from genplate import *
import model

num_label = 7
img_w = 120
img_h = 30
batch_size = 2
capacity = 200
count = 10

def rand_range(lo,hi):
    return lo+r(hi-lo)

def gen_rand():
    name = ""
    label=[]
    label.append(rand_range(0,31))
    label.append(rand_range(41,65))
    for i in range(5):
        label.append(rand_range(31,65))
        
    name+=chars[label[0]]
    name+=chars[label[1]]
    for i in range(5):
        name+=chars[label[i+2]]
    return name,label

def gen_sample(genplate,width, height):
    num,label =gen_rand()
    img = genplate.generate(num)
    img = cv2.resize(img,(width,height))
    img = np.multiply(img,1/255.0)
    #img = img.transpose(2,0,1)
    img = img.transpose(1,0,2)
    return label,img

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

class OCRIter():
    def __init__(self, count, batch_size, num_label, height, width):
        super(OCRIter, self).__init__()
        self.genplate = GenPlate("./font/platech.ttf",'./font/platechar.ttf','./NoPlates')
        self.batch_size = batch_size
        self.count = count
        self.height = height
        self.width = width
        print("make plate data")
     
    def iter(self):
        data = []
        label = []
        for i in range(self.batch_size):
            num, img = gen_sample(self.genplate, self.width, self.height)
            data.append(img)
            label.append(num)
        data_all = data
        label_all = label
        return data_all,label_all   
    
def ge_batch():
    data_batch = OCRIter(count,batch_size,num_label,img_h,img_w)
    image_batch,label_batch = data_batch.iter()
    label_batch1 = tf.convert_to_tensor(label_batch,tf.int32)
    label_batch2 = tf.reshape(tf.transpose(label_batch1),[-1]) 
    label_batch3 =tf.reshape(label_batch1,[-1])
    for i in range(len(image_batch)-1):
        image_batch1 = tf.convert_to_tensor(image_batch[i],tf.float32)
        #image_batch2 = tf.constant(0,shape=[120,30,3],dtype=tf.float32)
        #image_batch2 = tf.add(image_batch1,image_batch2)
        image_batch2 = tf.convert_to_tensor(image_batch[i+1],tf.float32)
        image_batch2 = tf.stack([image_batch1,image_batch2])
        
    return image_batch2,label_batch2,label_batch1,label_batch3,image_batch

def losses(logits, labels):
    '''Compute loss from logits and labels
    Args:
        logits: logits tensor, float, [batch_size, 65]
        labels: label tensor, tf.int32, [batch_size]
        
    Returns:
        loss tensor of float type
    '''
    with tf.variable_scope('loss') as scope:
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels, name='xentropy_per_example')
        loss = tf.reduce_mean(cross_entropy, name='loss')
        tf.summary.scalar(scope.name+'/loss', loss)
    return loss

img2,label2,label1,label0,img1= ge_batch()
train_logits = model.inference(img2,batch_size)
soft = tf.nn.softmax(train_logits)
loss  = losses(train_logits,label2)

sess = tf.Session()
with sess.as_default():
    sess.run(tf.global_variables_initializer())
    a = img2.eval()
    print(a)
    print(a.shape)
    print(img2.shape)
    print('####')
    e = train_logits.eval()
    print(e)
    print('###label2')
    b= label2.eval()
    print(b)
    print(b.shape)
    print(label2.shape)
    print('#loss')
    print(loss.eval())
    print('#soft')
    print(soft.eval())
    c= label1.eval()
    print(c)
    print(c.shape)
    print(label1.shape)
    d =label0.eval()
    print(d)
    print(d.shape)
    print(label0.shape)
    f =img1
    print(f)
    print('###')
    acc = model.evaluation(b,e)
    print(acc)

sess.close()

#%% test2
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  5 15:37:26 2017

@author: llc
"""

import os
import numpy as np
import tensorflow as tf
from input_data import OCRIter
import model
from genplate import *
import time
import datetime

num_label = 7
img_w = 272
img_h = 72
batch_size = 2
capacity = 200
count = 200
learning_rate = 0.0001

image_holder = tf.placeholder(tf.float32,[batch_size,img_w,img_h,3])
label_holder = tf.placeholder(tf.int32,[7*batch_size])

logs_train_dir = '/home/llc/TF_test/Chinese_plate_recognition/Plate_recognition/train_logs/'


#def get_batch():
#    data_batch = OCRIter(count,batch_size,num_label,img_h,img_w)
#    image_batch,label_batch = data_batch.iter()
#    label_batch1 = tf.convert_to_tensor(label_batch,tf.int32)
#    label_batch2 = tf.reshape(tf.transpose(label_batch1),[-1])   
#    for i in range(len(image_batch)-1):
#        image_batch1 = tf.convert_to_tensor(image_batch[i],tf.float32)
#        image_batch2 = tf.convert_to_tensor(image_batch[i+1],tf.float32)
#        image_batch2 = tf.stack([image_batch1,image_batch2])
#    return image_batch2,label_batch2

def get_batch():
    data_batch = OCRIter(count,batch_size,num_label,img_h,img_w)
    image_batch,label_batch = data_batch.iter()
    label_batch1 = np.reshape(np.transpose(label_batch),[-1])
    
    #label_batch1 = tf.one_hot(label_batch1,65)
    
    image_batch1 = np.array(image_batch)
    
    return image_batch1,label_batch1

                     

train_img_batch1,train_label_batch1 = get_batch()
train_logits = model.inference(image_holder,batch_size)
        
train_loss = model.losses(train_logits,label_holder) 
train_op = model.trainning(train_loss,learning_rate)

train_acc = model.evaluation(train_logits,label_holder)


summary_op = tf.summary.merge(tf.get_collection(tf.GraphKeys.SUMMARIES))
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
train_writer = tf.summary.FileWriter(logs_train_dir,sess.graph)
saver = tf.train.Saver()

sess.run(tf.global_variables_initializer())
      
for step in range(count): 
    start_time = time.time()
    time_str = datetime.datetime.now().isoformat()
    x_batch,y_batch = get_batch()
    feed_dict = {image_holder:x_batch,label_holder:y_batch}
    _,tra_loss,acc,summary_str= sess.run([train_op,train_loss,train_acc,summary_op],feed_dict)
   
    duration = time.time()-start_time
                        
    if step % 10 == 0:
        sec_per_batch = float(duration)
        print('%s : Step %d,train loss = %.2f,acc= %.2f,sec/batch=%.3f' %(time_str,step,tra_loss,acc,sec_per_batch))
        
        #print('Step %d,train loss = %.2f,train_accuracy=%.2f%%,sec/batch=%.3f' %(step,tra_loss,tra_acc*100.0,sec_per_batch))
        #summary_str = sess.run(summary_op)
        train_writer.add_summary(summary_str,step)
    if step % 5000 ==0 or (step+1) == count:
        checkpoint_path = os.path.join(logs_train_dir,'model.ckpt')
        saver = tf.train.Saver()
        saver.save(sess,checkpoint_path,global_step=step)
sess.close()       

#
#if __name__=='__main__':
#    run_training()



#%%  test 3
import numpy as np
import cv2
import os
from PIL import Image
import matplotlib.pyplot as plt

test_dir = '/home/llc/TF_test/Chinese_plate_recognition/Plate_recognition/plate/'
test_image = []
for file in os.listdir(test_dir):
    test_image.append(test_dir + file)

test_image = list(test_image)

n = len(test_image)
ind =np.random.randint(0,n)
img_dir = test_image[ind]
    
image = Image.open(img_dir)
plt.imshow(image)
image = image.resize([120,30])
image = np.multiply(image,1/255.0)
print(image.shape) 
image = np.array(image)
print(image)
print(image.shape)
image1 = image.transpose(1,0,2)
print(image1.shape)
#%%  Valid
import tensorflow as tf
import numpy as np
import os
import cv2
from PIL import Image
import matplotlib.pyplot as plt
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
         

#### test tf.image.decode_jpeg & cv2.read
img = tf.read_file('./plate/01.jpg')
img = tf.image.decode_jpeg(img,channels=3)
img = tf.multiply(img,int(1/255))
print(img.shape)
####

#img = cv2.imread('./plate/01.jpg')
#img = np.multiply(img,1/255.0)
#image = np.array([img])
#print (image.shape)

batch_size = 1
x = tf.placeholder(tf.float32,[batch_size,72,272,3])


logit = model.inference(x,batch_size)
logit1 = tf.reshape(tf.nn.softmax(logit),[-1,65])

   
logs_train_dir = '/home/llc/TF_test/Chinese_plate_recognition/Plate_recognition/train_logs_50000/'

saver = tf.train.Saver()

with tf.Session() as sess:
    print ("Reading checkpoint...")
    ckpt = tf.train.get_checkpoint_state(logs_train_dir)
    if ckpt and ckpt.model_checkpoint_path:
        global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
        saver.restore(sess, ckpt.model_checkpoint_path)
        print('Loading success, global_step is %s' % global_step)
    else:
        print('No checkpoint file found')
        
    ### test tf.image.decode_jpeg & cv2.read
    image = sess.run(img)
    image=[image]
    ###
    prediction = sess.run(logit1, feed_dict={x: image})

#    print(prediction)
    max_index = np.argmax(prediction,axis=1)
    print(max_index)
    line = ''
    for i in range(prediction.shape[0]):
        if i == 0:
            result = np.argmax(prediction[i][0:31])
        if i == 1:
            result = np.argmax(prediction[i][41:65])+41
        if i > 1:
            result = np.argmax(prediction[i][31:65])+31
        
        line += chars[result]+" "
    print ('predicted: ' + line)  


#%%  Valid 2

import tensorflow as tf
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt
from input_data import gen_rand,gen_sample
from input_data import OCRIter
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
batch_size = 2
img_w = 272
img_h = 72
capacity = 200

x = tf.placeholder(tf.float32,[batch_size,72,272,3])
keep_prob =tf.placeholder(tf.float32)
       
data_batch = OCRIter(batch_size,img_h,img_w)
image_batch,label_batch = data_batch.iter()
 
image_batch1 = np.array(image_batch)        
         
print(image_batch1.shape)


logit1,logit2,logit3,logit4,logit5,logit6,logit7 = model.inference(x,keep_prob)
#logit1 = tf.nn.softmax(logit)

   
logs_train_dir = '/home/llc/TF_test/Chinese_plate_recognition/Plate_recognition/train_logs_50000/'

saver = tf.train.Saver()

with tf.Session() as sess:
    print ("Reading checkpoint...")
    ckpt = tf.train.get_checkpoint_state(logs_train_dir)
    if ckpt and ckpt.model_checkpoint_path:
        global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
        saver.restore(sess, ckpt.model_checkpoint_path)
        print('Loading success, global_step is %s' % global_step)
    else:
        print('No checkpoint file found')

    pre1,pre2,pre3,pre4,pre5,pre6,pre7 = sess.run([logit1,logit2,logit3,logit4,logit5,logit6,logit7], feed_dict={x: image_batch1,keep_prob:1})
    prediction = np.reshape(np.array([pre1,pre2,pre3,pre4,pre5,pre6,pre7]),[-1,65])
#    print(prediction)
    max_index = np.argmax(prediction,axis=1)
    print(max_index)
    line = ''
    for i in range(prediction.shape[0]):
        if i == 0:
            result = np.argmax(prediction[i][0:31])
        if i == 1:
            result = np.argmax(prediction[i][41:65])+41
        if i > 1:
            result = np.argmax(prediction[i][31:65])+31
        
        line += chars[result]+" "
    print ('predicted: ' + line)  
    print(label_batch)
#%% test cv2.imread
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2
from input_data import OCRIter

#data_batch = OCRIter(batch_size,img_h,img_w)
#image_batch,label_batch = data_batch.iter() 
#image_batch1 = np.array(image_batch)       
#outputPath="./plate"

img0 = cv2.imread('./plate/00.jpg')
#img = Image.open('./plate/00.jpg')
#img = img.resize([120,30])
#plt.imshow(img)

#img = np.array(img0)
#cv2.imwrite(outputPath + "/" +"1"+".jpg", img)
#print(img.shape)
#print(img)

img = np.multiply(img0,1/255.0)
print(img.shape)
image = img.transpose(1,0,2)

#image0 = np.reshape(image,[1,120,30,3])
image1 = np.array([image])
print(image1.shape)
#%%  test tf.stack
 

import tensorflow as tf

a =tf.constant([[1,2],[3,4]])

b =tf.constant([[5,6],[7,8]])

c = tf.constant([[9,10],[10,11]])

d = tf.stack([a,b,c],axis=0)

e =tf.stack([a,b,c],axis=1)

f = tf.stack([a,b,c],axis=2)

with tf.Session() as sess:
    print(sess.run(d))
    print('###')
    print(sess.run(e))
    print('###')
    print(sess.run(f))
