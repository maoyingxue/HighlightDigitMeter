# -*- coding: utf-8 -*-
"""
Created on Fri Jan 11 13:28:46 2019

@author: maoyingxue
"""
import tensorflow as tf
import os
import numpy as np
import cv2
files=os.listdir("train_data/")
print(len(files))
imgs=[]
ys=[]
for file in files:
    img=cv2.imread("train_data/"+file)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 127,255,cv2.THRESH_BINARY) 
    
    thresh=thresh.reshape((784)).tolist()
    thresh=np.minimum(thresh,1)
    imgs.append(thresh)
    if file[0]=="n":
        ys.append(10)
    else:
        ys.append(int(file[0]))
imgs=np.array(imgs,dtype="float")
labels=np.zeros((len(ys),11))
for i,y in enumerate(ys):
    labels[i][y]=1

#daluan shuju
permutation = np.random.permutation(labels.shape[0])
imgs = imgs[permutation, :]
labels = labels[permutation,:]
print(imgs[0],labels[0])

#train data test data
train_imgs=imgs[:-100]
train_labels=labels[:-100]
test_imgs=imgs[-100:]
test_labels=labels[-100:]
# 构建图
sess = tf.InteractiveSession()
x = tf.placeholder(tf.float32, [None, 784],name="img")
W = tf.Variable(tf.zeros([784,11]),name="w")
b = tf.Variable(tf.zeros([11]),name="b")

y = tf.nn.softmax(tf.matmul(x,W) + b)
predict=tf.argmax(y,1,name="predict")
y_ = tf.placeholder(tf.float32, [None,11])
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y),reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

# 进行训练
tf.global_variables_initializer().run()
saver=tf.train.Saver()
j=0
batch=20
total=train_imgs.shape[0]
print(total)
for i in range(10000):
  #batch_xs = train_imgs[(i*batch)%total:((i+1)*batch)%total]
  #batch_ys=train_labels[(i*batch)%total:((i+1)*batch)%total]
  batch_xs=train_imgs[(i*batch)%total:((i+1)*batch)%total].reshape(-1,784)
  batch_ys=train_labels[(i*batch)%total:((i+1)*batch)%total].reshape(-1,11)
  #print(batch_xs,batch_ys)
  train_step.run({x: batch_xs, y_: batch_ys})
  j=j+1
saver.save(sess,'checkpoint/model')
# 模型评估
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

print('MNIST手写图片准确率')
print(accuracy.eval({x: test_imgs, y_: test_labels}))