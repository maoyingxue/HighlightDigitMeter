# -*- coding: utf-8 -*-
"""
Created on Fri Jan 11 14:25:04 2019

@author: maoyingxue
"""

import cv2
import tensorflow as tf
import numpy as np
import os

#test
sess=tf.Session()
saver = tf.train.import_meta_graph('./checkpoint2/model.meta')
saver.restore(sess, tf.train.latest_checkpoint('./checkpoint2'))
graph = tf.get_default_graph()
image=graph.get_tensor_by_name("img:0")
predict=graph.get_tensor_by_name("predict:0")
prob=graph.get_tensor_by_name("prob:0")
def test(thresh):
    prediction=sess.run(predict,feed_dict={image:thresh,prob: 1.0})
    print("predict number:",prediction[0])
if __name__ == '__main__':
    files=os.listdir("train_data/")
    for file in files:
        img = cv2.imread("train_data/" + file)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

        thresh = thresh.reshape((-1,784)).tolist()
        thresh = np.minimum(thresh, 1)
        print(file)
        #print(thresh)
        test(thresh)
       # break



