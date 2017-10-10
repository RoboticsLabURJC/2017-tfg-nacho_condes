#
# Created on Mar 7, 2017
#
# @author: dpascualhe
#
# Based on @nuriaoyaga code:
# https://github.com/RoboticsURJC-students/2016-tfg-nuria-oyaga/blob/
#     master/camera/camera.py
#
# And @Javii91 code:
# https://github.com/Javii91/Domotic/blob/master/Others/cameraview.py
#

import os
import sys
import random
import traceback
import threading

import cv2
import numpy as np
import easyiceconfig as EasyIce
from PIL import Image
from jderobot import CameraPrx
import tensorflow as tf


class Camera:

    def __init__ (self):
        ''' Camera class gets images from live video and transform them
        in order to predict the digit in the image.
        '''
        status = 0
        ic = None
        
        # Initializing the Ice run-time.
        ic = EasyIce.initialize(sys.argv)
        properties = ic.getProperties()
        self.lock = threading.Lock()

    
        try:
            # We obtain a proxy for the camera.
            #obj = ic.propertyToProxy("Digitclassifier.Camera.Proxy")
	    obj = ic.stringToProxy('cameraA:default -h localhost -p 9999')
            # We get the first image and print its description.
            self.cam = CameraPrx.checkedCast(obj)

            if self.cam:
                self.im = self.cam.getImageData("RGB8")
                self.im_height = self.im.description.height
                self.im_width = self.im.description.width
                print(self.im.description)
            else: 
                print("Interface camera not connected")
                    
        except:
            traceback.print_exc()
            exit()
            status = 1

    def getImage(self):
        ''' Gets the image from the webcam and returns the original
        image with a ROI draw over it and the transformed image that
        we're going to use to make the prediction.
        '''      
        if self.cam:            
            self.lock.acquire()
            
            im = np.zeros((self.im_height, self.im_width, 3), np.uint8)
            im = np.frombuffer(self.im.pixelData, dtype=np.uint8)
            im.shape = self.im_height, self.im_width, 3
            im_trans = self.trasformImage(im)
            # It prints the ROI over the live video
            cv2.rectangle(im, (218, 138), (422, 342), (0, 0, 255), 2)
            ims = [im, im_trans]
            
            self.lock.release()
            
            return ims
    
    def update(self):
        ''' Updates the camera every time the thread changes. '''
        if self.cam:
            self.lock.acquire()
            
            self.im = self.cam.getImageData("RGB8")
            self.im_height = self.im.description.height
            self.im_width = self.im.description.width
            
            self.lock.release()

    def trasformImage(self, im):
        ''' Transforms the image into a 28x28 pixel grayscale image and
        applies a sobel filter (both x and y directions).
        ''' 
        im_crop = im [140:340, 220:420]
        im_gray = cv2.cvtColor(im_crop, cv2.COLOR_BGR2GRAY)
        im_blur = cv2.GaussianBlur(im_gray, (5, 5), 0) # Noise reduction.
                   
        im_res = cv2.resize(im_blur, (28, 28))

        # Edge extraction.
        im_sobel_x = cv2.Sobel(im_res, cv2.CV_32F, 1, 0, ksize=5)
        im_sobel_y = cv2.Sobel(im_res, cv2.CV_32F, 0, 1, ksize=5)
        im_edges = cv2.add(abs(im_sobel_x), abs(im_sobel_y))
        im_edges = cv2.normalize(im_edges, None, 0, 255, cv2.NORM_MINMAX)
        im_edges = np.uint8(im_edges)
        
        return im_edges

    
    def classification(self, im):

        reshaped_img = np.reshape(im, (1,784))

        # TF variables
        x = tf.placeholder(tf.float32, shape=[None, 784])
        y_ = tf.placeholder(tf.float32, shape=[None, 10])

        W = tf.Variable(tf.zeros([784,10]))
        b = tf.Variable(tf.zeros([10]))

        y = tf.nn.softmax(tf.matmul(x, W) + b)

        saver = tf.train.Saver()

        with tf.Session() as sess:
            saver.restore(sess, './networks/my-model/model-900')
            probs = sess.run(y, feed_dict={x: reshaped_img})
            prediction = probs.argmax()

        tf.reset_default_graph()

        return prediction
    
