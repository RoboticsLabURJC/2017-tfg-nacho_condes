#
# Created on Jan 18, 2018
#
# @author: naxvm
#
# Based on @nuriaoyaga code:
# https://github.com/RoboticsURJC-students/2016-tfg-nuria-oyaga/blob/
#     master/gui/gui.py
#


from PyQt5 import QtGui
from PyQt5 import QtCore
from PyQt5 import QtWidgets


import numpy as np
import cv2

from cprint import *


class GUI(QtWidgets.QWidget):

    updGUI = QtCore.pyqtSignal()

    def __init__(self, parent=None):
        ''' GUI class creates the GUI that we're going to use to
        preview the live video as well as the results of the real-time
        classification.
        '''
        QtWidgets.QWidget.__init__(self, parent)
        self.resize(1200, 500)
        self.move(150, 50)
        self.setWindowIcon(QtGui.QIcon('resources/jderobot.png'))
        self.updGUI.connect(self.update)

        # Original image label.
        self.im_label = QtWidgets.QLabel(self)
        self.im_label.resize(450, 350)
        self.im_label.move(25, 90)
        self.im_label.show()

        # Video capture framerate label.
        self.video_framerate_label = QtWidgets.QLabel(self)
        self.video_framerate_label.move(220, 450)
        self.video_framerate_label.resize(50, 40)
        self.video_framerate_label.show()

        # Processed image label.
        self.im_pred_label = QtWidgets.QLabel(self)
        self.im_pred_label.resize(450, 350)
        self.im_pred_label.move(725, 90)
        self.im_pred_label.show()

        # Prediction framerate label.
        self.predict_framerate_label = QtWidgets.QLabel(self)
        self.predict_framerate_label.move(930, 450)
        self.predict_framerate_label.resize(50,40)
        self.predict_framerate_label.show()

        # Button for configuring detection flow
        self.button_cont_detection = QtWidgets.QPushButton(self)
        self.button_cont_detection.move(550, 100)
        self.button_cont_detection.clicked.connect(self.toggleNetwork)
        self.button_cont_detection.setText('Continuous')
        self.button_cont_detection.setStyleSheet('QPushButton {color: green;}')

        # Button for processing a single frame
        self.button_one_frame = QtWidgets.QPushButton(self)
        self.button_one_frame.move(555, 200)
        self.button_one_frame.clicked.connect(self.updateOnce)
        self.button_one_frame.setText('Step')

        # Logo
        self.logo_label = QtWidgets.QLabel(self)
        self.logo_label.resize(150, 150)
        self.logo_label.move(520, 300)
        self.logo_label.setScaledContents(True)

        logo_img = QtGui.QImage()
        logo_img.load('resources/jderobot.png')
        self.logo_label.setPixmap(QtGui.QPixmap.fromImage(logo_img))
        self.logo_label.show()

        self.font = cv2.FONT_HERSHEY_PLAIN
        self.scale = 2.0


    def setCamera(self, cam, t_cam):
        ''' Declares the Camera object '''
        self.cam = cam
        self.t_cam = t_cam


    def setNetwork(self, network, t_network):
        ''' Declares the Network object and its corresponding control thread. '''
        self.network = network

        if self.network.framework == "TensorFlow":
            self.setWindowTitle("JdeRobot-TensorFlow tracker")
        else:
            self.setWindowTitle("JdeRobot-Keras tracker")

        self.t_network = t_network


    def setMotors(self, motors, t_motors):
        self.motors = motors
        self.t_motors = t_motors


    def update(self):
        '''
        Updates the GUI for every time the thread changes.
        '''
        # We get the original image and display it.
        self.im_prev = self.cam.getImage()
        im = QtGui.QImage(self.im_prev.data, self.im_prev.shape[1], self.im_prev.shape[0],
                          QtGui.QImage.Format_RGB888)
        self.im_scaled = im.scaled(self.im_label.size())

        self.im_label.setPixmap(QtGui.QPixmap.fromImage(self.im_scaled))

        if self.t_network.is_activated:
            self.renderModifiedImage()

        self.predict_framerate_label.setText("%d fps" % (self.t_network.framerate))
        self.video_framerate_label.setText("%d fps" % (self.t_cam.framerate))


    def toggleNetwork(self):
        self.t_network.toggle()
        self.t_motors.toggle()

        if self.t_network.is_activated:
            self.button_cont_detection.setStyleSheet('QPushButton {color: green;}')
        else:
            self.button_cont_detection.setStyleSheet('QPushButton {color: red;}')

    def updateOnce(self):
        self.t_network.runOnce()
        self.t_motors.runOnce()
        self.renderModifiedImage()


    def renderModifiedImage(self, img_dest='rgb'):
        if img_dest == 'rgb':
            image_np = np.copy(self.im_prev)
        else:
            image_np = np.copy(self.depth_prev)

        detected_persons = self.motors.persons
        detected_faces = self.motors.faces

        if img_dest == 'rgb':
            for face in detected_faces:
                # We draw the faces on the image
                xmin = face[0]
                ymin = face[1]
                xmax = face[2]
                ymax = face[3]
                cv2.rectangle(image_np, (xmin, ymax), (xmax, ymin), (255, 0, 0), 2)


        img_shape = image_np.shape

        for index in range(len(detected_persons)):
            person = detected_persons[index]
            score = person.score

            [xmin, ymin, xmax, ymax] = person

            if person.is_mom:
                # This rect belongs to mom
                cv2.rectangle(image_np, (xmin, ymax), (xmax, ymin), (0,255,0), 5)
                label = "MOM ({} %)".format(int(score*100))
            else:
                cv2.rectangle(image_np, (xmin, ymax), (xmax, ymin), (255,255,0), 2)
                label = "person ({} %)".format(int(score*100))


            [size, base] = cv2.getTextSize(label, self.font, self.scale, 3)
            xmin = max(xmin, 0)
            ymin = max(ymin, base + ymin + size[1])

            points = np.array([[[xmin, ymin + base],
                                [xmin, ymin - size[1]],
                                [xmin + size[0], ymin - size[1]],
                                [xmin + size[0], ymin + base]]], dtype=np.int32)
            cv2.fillPoly(image_np, points, (0, 0, 0))
            cv2.putText(image_np, label, (xmin, ymin), self.font, self.scale, (255, 255, 255), 2)

        im = QtGui.QImage(image_np.data, img_shape[1], img_shape[0],
                          QtGui.QImage.Format_RGB888)

        im_drawn = im.scaled(self.im_label.size())
        if img_dest == 'rgb':
            self.im_pred_label.setPixmap(QtGui.QPixmap.fromImage(im_drawn))
        else:
            self.depth_label.setPixmap(QtGui.QPixmap.fromImage(im_drawn))


class DepthGUI(GUI):
    ''' This class inherits all the standard GUI behavioral, except the
    configuration to additionally show the depth image from the XTION sensor. '''

    def __init__(self):
        # Standard initialization
        GUI.__init__(self)
        # Now, we add/move labels to display the depth image
        self.resize(1300, 700)
        self.im_label.move(25, 150)

        self.video_framerate_label.move(220, 500)

        self.im_pred_label.resize(450, 320)
        self.im_pred_label.move(780, 10)
        self.predict_framerate_label.move(980, 320)

        # New label (for the depth map)
        self.depth_label = QtWidgets.QLabel(self)
        self.depth_label.move(780, 370)
        self.depth_label.resize(450, 320)
        self.depth_label.show()

    def setDepth(self, depth, t_depth):
        self.depth = depth
        self.t_depth = t_depth


    def setMotors(self, motors, t_motors):
        self.motors = motors
        self.t_motors = t_motors


    def renderModifiedImage(self):
        GUI.renderModifiedImage(self)
        GUI.renderModifiedImage(self, 'depth')

    def update(self):
        depth_total = self.depth.getImage()
        layers = cv2.split(depth_total)
        self.depth_prev = cv2.applyColorMap(cv2.cvtColor(layers[0], cv2.COLOR_GRAY2BGR), cv2.COLORMAP_JET)

        depth = QtGui.QImage(self.depth_prev, self.depth_prev.shape[0], self.depth_prev.shape[1],
                             QtGui.QImage.Format_RGB888)
        self.depth_scaled = depth.scaled(self.depth_label.size())
        self.depth_label.setPixmap(QtGui.QPixmap.fromImage(self.depth_scaled))

        GUI.update(self)
