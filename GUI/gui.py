#
# Created on Mar 7, 2017
#
# @author: dpascualhe
#
# Based on @nuriaoyaga code:
# https://github.com/RoboticsURJC-students/2016-tfg-nuria-oyaga/blob/
#     master/gui/gui.py
#

import sys

from PyQt5 import QtGui
from PyQt5 import QtCore
from PyQt5 import QtWidgets

from Net.threadnetwork import ThreadNetwork


class GUI(QtWidgets.QWidget):

    updGUI = QtCore.pyqtSignal()

    def __init__(self, parent=None):
        ''' GUI class creates the GUI that we're going to use to
        preview the live video as well as the results of the real-time
        classification.
        '''

        QtWidgets.QWidget.__init__(self, parent)
        self.setWindowTitle("Digit Classification (TensorFlow CNN trained " +
                            "with a MNIST database)")
        self.setWindowIcon(QtGui.QIcon('resources/jderobot.png'))
        self.resize(1200, 600)
        self.move(150, 50)
        self.updGUI.connect(self.update)

        # Original image label.
        self.im_label = QtWidgets.QLabel(self)
        self.im_label.resize(500, 400)
        self.im_label.move(70, 50)
        self.im_label.show()

        # Transformed image label.
        self.im_trans_label = QtWidgets.QLabel(self)
        self.im_trans_label.resize(200, 200)
        self.im_trans_label.move(900, 50)
        self.im_trans_label.show()


        self.dgts = []
        # Digit labels from 0 to 9.
        lab0 = QtWidgets.QLabel(self)
        lab0.resize(30, 30)
        lab0.move(985, 450)
        lab0.setText('0')
        lab0.setAlignment(QtCore.Qt.AlignCenter)
        lab0.setStyleSheet("background-color: #7FFFD4; color: #000; " +
                           "font-size: 20px; border: 1px solid black;")
        self.dgts.append(lab0)

        lab1 = QtWidgets.QLabel(self)
        lab1.resize(30, 30)
        lab1.move(900, 300)
        lab1.setText('1')
        lab1.setAlignment(QtCore.Qt.AlignCenter)
        lab0.setStyleSheet("background-color: #7FFFD4; color: #000; " +
                           "font-size: 20px; border: 1px solid black;")
        self.dgts.append(lab1)

        lab2 = QtWidgets.QLabel(self)
        lab2.resize(30, 30)
        lab2.move(985, 300)
        lab2.setText('2')
        lab2.setAlignment(QtCore.Qt.AlignCenter)
        lab0.setStyleSheet("background-color: #7FFFD4; color: #000; " +
                           "font-size: 20px; border: 1px solid black;")
        self.dgts.append(lab2)

        lab3 = QtWidgets.QLabel(self)
        lab3.resize(30, 30)
        lab3.move(1070, 300)
        lab3.setText('3')
        lab3.setAlignment(QtCore.Qt.AlignCenter)
        lab0.setStyleSheet("background-color: #7FFFD4; color: #000; " +
                           "font-size: 20px; border: 1px solid black;")
        self.dgts.append(lab3)

        lab4 = QtWidgets.QLabel(self)
        lab4.resize(30, 30)
        lab4.move(900, 350)
        lab4.setText('4')
        lab4.setAlignment(QtCore.Qt.AlignCenter)
        lab0.setStyleSheet("background-color: #7FFFD4; color: #000; " +
                           "font-size: 20px; border: 1px solid black;")
        self.dgts.append(lab4)

        lab5 = QtWidgets.QLabel(self)
        lab5.resize(30, 30)
        lab5.move(985, 350)
        lab5.setText('5')
        lab5.setAlignment(QtCore.Qt.AlignCenter)
        lab0.setStyleSheet("background-color: #7FFFD4; color: #000; " +
                           "font-size: 20px; border: 1px solid black;")
        self.dgts.append(lab5)

        lab6 = QtWidgets.QLabel(self)
        lab6.resize(30, 30)
        lab6.move(1070, 350)
        lab6.setText('6')
        lab6.setAlignment(QtCore.Qt.AlignCenter)
        lab0.setStyleSheet("background-color: #7FFFD4; color: #000; " +
                           "font-size: 20px; border: 1px solid black;")
        self.dgts.append(lab6)

        lab7 = QtWidgets.QLabel(self)
        lab7.resize(30, 30)
        lab7.move(900, 400)
        lab7.setText('7')
        lab7.setAlignment(QtCore.Qt.AlignCenter)
        lab0.setStyleSheet("background-color: #7FFFD4; color: #000; " +
                           "font-size: 20px; border: 1px solid black;")
        self.dgts.append(lab7)

        lab8 = QtWidgets.QLabel(self)
        lab8.resize(30, 30)
        lab8.move(985, 400)
        lab8.setText('8')
        lab8.setAlignment(QtCore.Qt.AlignCenter)
        lab0.setStyleSheet("background-color: #7FFFD4; color: #000; " +
                           "font-size: 20px; border: 1px solid black;")
        self.dgts.append(lab8)

        lab9 = QtWidgets.QLabel(self)
        lab9.resize(30, 30)
        lab9.move(1070, 400)
        lab9.setText('9')
        lab9.setAlignment(QtCore.Qt.AlignCenter)
        lab0.setStyleSheet("background-color: #7FFFD4; color: #000; " +
                           "font-size: 20px; border: 1px solid black;")
        self.dgts.append(lab9)



         # Button for configuring detection flow
        self.button_cont_detection = QtWidgets.QPushButton(self)
        self.button_cont_detection.move(700, 100)
        self.button_cont_detection.clicked.connect(self.toggleNetwork)


        # Button for processing a single frame
        self.button_one_frame = QtWidgets.QPushButton(self)
        self.button_one_frame.move(700, 200)
        self.button_one_frame.clicked.connect(self.updateOnce)
        self.button_one_frame.setText('On-demand\ndetection')

        # Logo
        self.logo_label = QtWidgets.QLabel(self)
        self.logo_label.resize(150, 150)
        self.logo_label.move(660, 300)
        self.logo_label.setScaledContents(True)

        logo_img = QtGui.QImage()
        logo_img.load('GUI/resources/jderobot.png')
        self.logo_label.setPixmap(QtGui.QPixmap.fromImage(logo_img))
        self.logo_label.show()




    def setCamera(self, cam):
        ''' Declares the Camera object '''
        self.cam = cam
        # Network initialization.
        
        self.t_network = ThreadNetwork(self.cam.network)
        self.t_network.start()
        self.toggleNetwork()

    def update(self):
        ''' Updates the GUI for every time the thread change '''
        # We get the original image and display it.
        im_prev = self.cam.getImage()[0]
        im = QtGui.QImage(im_prev.data, im_prev.shape[1], im_prev.shape[0],
                          QtGui.QImage.Format_RGB888)
        im_scaled = im.scaled(self.im_label.size())
        self.im_label.setPixmap(QtGui.QPixmap.fromImage(im_scaled))



        if self.t_network.activated:
            self.updateTransImg()
            
        # We "turn on" the digit that it's been classified.
        self.lightON(self.cam.network.output_digit)

    def lightON(self, out):
        ''' Updates which digit has the "light on" depending on the
        network output.
        '''

        for dgt in self.dgts:
            dgt.setStyleSheet("background-color: #7FFFD4; color: #000; " +
                              "font-size: 20px; border: 1px solid black;")
            if out is not None:
                self.dgts[out].setStyleSheet("background-color: #FFFF00; " +
                                             "color: #000; font-size: 20px; " +
                                             "border: 1px solid black;")

    def toggleNetwork(self):
        self.t_network.activated = not self.t_network.activated

        if self.t_network.activated:
            self.button_cont_detection.setStyleSheet('QPushButton {color: red;}')
            self.button_cont_detection.setText('Switch off\nContinuous\nDetection')
        else:
            self.button_cont_detection.setStyleSheet('QPushButton {color: green;}')
            self.button_cont_detection.setText('Switch on\nContinuous\nDetection')



    def updateTransImg(self):

        # We get the transformed image and display it.
        im_prev_trans = self.cam.getImage()[1]

        self.cam.network.input_image = im_prev_trans

        im_trans = QtGui.QImage(im_prev_trans.data, im_prev_trans.shape[1],
                                    im_prev_trans.shape[0],
                                    QtGui.QImage.Format_Indexed8)
        im_trans_cvt = im_trans.convertToFormat(QtGui.QImage.Format_Indexed8)
        colortable = [QtGui.qRgb(i, i, i) for i in range(255)]
        im_trans.setColorTable(colortable)
        im_trans_scaled = im_trans.scaled(self.im_trans_label.size())
        self.im_trans_label.setPixmap(QtGui.QPixmap.fromImage(im_trans_scaled))


    def updateOnce(self):
        self.t_network.runOnce()
        self.updateTransImg()