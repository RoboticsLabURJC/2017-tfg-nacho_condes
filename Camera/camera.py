#
# Created on Oct, 2017
#
# @author: naxvm
#
# Class which abstracts a Camera from a proxy (created by ICE/ROS),
# and provides the methods to keep it constantly updated. Also, it processes
# it by using a Sobel edges filter, and delivers it to the neural network,
# which returns the predicted digit.
#
#
# Based on @nuriaoyaga code:
# https://github.com/RoboticsURJC-students/2016-tfg-nuria-oyaga/blob/
#     master/numberclassifier.py
# and @dpascualhe's:
# https://github.com/RoboticsURJC-students/2016-tfg-david-pascual/blob/
#     master/digitclassifier.py
#
#

import os
import sys
import random
import traceback
import threading

import cv2
import numpy as np
from PIL import Image
from jderobot import CameraPrx
import tensorflow as tf
from Net.network import Network
import comm
import config


class Camera:

    def __init__(self):
        ''' Camera class gets images from live video and transform them
        in order to predict the digit in the image.
        '''
        status = 0

        # Creation of the camera through the comm-ICE proxy.
        try:
            cfg = config.load(sys.argv[1])
        except IndexError:
            raise SystemExit('Error: Missing YML file. \n  Usage: python2 digitclassifier.py digitclassifier.yml')

        jdrc = comm.init(cfg, 'DigitClassifier')

        self.lock = threading.Lock()

        # Creation of the network, and load of the model into it.
        self.network = Network('Net/mnist-model')

        try:
            
            self.cam = jdrc.getCameraClient('DigitClassifier.Camera')
            if self.cam:
                self.im = self.cam.getImage()
                self.im_height = self.im.height
                self.im_width = self.im.width
                print('Image size: {0}x{1} px'.format(
                        self.im_width, self.im_height))
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
            im = np.frombuffer(self.im.data, dtype=np.uint8)
            im.shape = self.im_height, self.im_width, 3
            im_trans = self.transformImage(im)

            cv2.rectangle(im, (218, 138), (422, 342), (0, 0, 255), 2)
            ims = [im, im_trans]

            self.lock.release()

            return ims

    def update(self):
        ''' Updates the camera every time the thread changes. '''
        if self.cam:
            self.lock.acquire()

            self.im = self.cam.getImage()
            self.im_height = self.im.height
            self.im_width = self.im.width

            self.lock.release()

    def transformImage(self, im):
        ''' Transforms the image into a 28x28 pixel grayscale image and
        applies a sobel filter (both x and y directions).
        '''
        im_crop = im[140:340, 220:420]
        im_gray = cv2.cvtColor(im_crop, cv2.COLOR_BGR2GRAY)
        im_blur = cv2.GaussianBlur(im_gray, (5, 5), 0)  # Noise reduction.

        im_res = cv2.resize(im_blur, (28, 28))

        # Edge extraction.
        im_sobel_x = cv2.Sobel(im_res, cv2.CV_32F, 1, 0, ksize=5)
        im_sobel_y = cv2.Sobel(im_res, cv2.CV_32F, 0, 1, ksize=5)
        im_edges = cv2.add(abs(im_sobel_x), abs(im_sobel_y))
        im_edges = cv2.normalize(im_edges, None, 0, 255, cv2.NORM_MINMAX)
        im_edges = np.uint8(im_edges)

        return im_edges

    def classification(self, im):
        ''' Calls the prediction method, and returns the digit
        which the neural network yields.'''
        prediction = self.network.classify(im).argmax()

        return prediction
