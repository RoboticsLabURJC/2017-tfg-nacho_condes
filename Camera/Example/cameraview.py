'''
Created on Mar 1, 2017

@author: dpascualhe

It opens a camera connection and converts the data received into displayable
images that are shown and updated in a window.

Based on @Javii91 code:
https://github.com/Javii91/Domotic/blob/master/Others/cameraview.py

And @nuriaoyaga code:
https://github.com/RoboticsURJC-students/2016-tfg-nuria-oyaga/blob/master/cameraview.py

'''

from PIL import Image
from jderobot import CameraPrx
import Ice, traceback
import cv2
import numpy
import sys


class Cameraview:
        
    # It converts bytes that contain image data into an understable image
    def data_to_image (self, data):
        img = Image.frombytes('RGB', (data.description.width,
                                      data.description.height), data.pixelData,
                                 'raw', "BGR")
        pix = numpy.array(img)
        return pix
    
    # It crops an image region and apply a morphological gradient
    def process_img (self, img):
        kernel = numpy.ones((3, 3))
        cv2.rectangle(img, (218, 138), (422, 342), (0, 0, 255), 2)
        img_crop = img [140:340, 220:420]
        
        img_bw = cv2.cvtColor(img_crop, cv2.COLOR_BGR2GRAY)
        img_dil = cv2.dilate(img_bw, kernel)
        img_ero = cv2.erode(img_bw, kernel)
        img_mgrad = img_dil - img_ero
        
        return img_mgrad
        
    def __init__ (self):
                
        status = 0
        ic = None
        print "hi"
    
        try:        
            # Initializing the Ice run time
            ic = Ice.initialize()
            # Obtaining a proxy for the camera (obj. identity: address)
            obj = ic.stringToProxy('cameraA:default -h localhost -p 9999')
            
            # We get the first image and print its description
            cam = CameraPrx.checkedCast(obj)
            data = cam.getImageData("RGB8")
            if cam:
                print(data.description)
            else:
                print "no cam"
            
            # We define the windows that we're going to employ
            cv2.namedWindow("Live video")
            cv2.namedWindow("Processed image")
            cv2.moveWindow("Live video", 60, 60)
            cv2.moveWindow("Processed image", 800, 200)
            
            while (1):
                # It checks and returns the proxy
                cam = CameraPrx.checkedCast(obj)
                # We get the bytes that contain the image data and convert them 
                # into an understandable image
                data = cam.getImageData("RGB8")
                img = self.data_to_image(data)
                # We process the image and display it
                img_proc = self.process_img(img)
                cv2.imshow("Live video", img)
                cv2.imshow("Processed image", img_proc)
                cv2.waitKey(25)
                
            cv2.destroyAllWindows()
    
        except:
            traceback.print_exc()
            status = 1    
        
        if ic:
            
            try:
                ic.destroy()
            
            except:
                traceback.print_exc()
                status = 1
    
        sys.exit(status)
