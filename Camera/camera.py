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

import traceback
import threading

import cv2
import numpy as np

class Camera:

    def __init__(self, proxy):
        ''' Camera class gets images from live video and transform them
        in order to predict the digit in the image.
        '''

        self.lock = threading.Lock()


        try:

            self.cam = proxy
        except:
            traceback.print_exc()
            exit()

    def get_image(self):
        ''' Gets the image from the webcam and returns the original
        image with a ROI draw over it and the transformed image that
        we're going to use to make the prediction.
        '''
        if self.cam:
            self.lock.acquire()

            im = np.frombuffer(self.im.data, dtype=np.uint8)
            im.shape = self.im.height, self.im.width, 3

            cv2.rectangle(im, (218, 138), (422, 342), (0, 0, 255), 2)

            self.lock.release()

            return im

    def update(self):
        ''' Updates the camera every time the thread changes. '''
        if self.cam:
            self.lock.acquire()

            self.im = self.cam.getImage()

            self.lock.release()
