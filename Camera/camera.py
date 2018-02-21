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
        self.model_path = cfg.getNode()['DigitClassifier']['Model']

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

            cv2.rectangle(im, (218, 138), (422, 342), (0, 0, 255), 2)

            self.lock.release()

            return im

    def update(self):
        ''' Updates the camera every time the thread changes. '''
        if self.cam:
            self.lock.acquire()

            self.im = self.cam.getImage()
            self.im_height = self.im.height
            self.im_width = self.im.width

            self.lock.release()

    def classification(self, im):
        ''' Calls the prediction method, and returns the digit
        which the neural network yields.'''
        prediction = self.network.classify(im).argmax()

        return prediction
