#
# Created on May, 2018
#
# @author: naxvm
#
# Based on dl-objectdetector
# https://github.com/jderobot/dl-objectdetector

import sys
import signal
# import comm

# from PyQt5 import QtWidgets

from Camera.ROSCam import ROSCam
from Net.TensorFlow.network import TrackingNetwork
from GUI.threadgui import ThreadGUI
from Net.threadnetwork import ThreadNetwork
from Motors.threadmotors import ThreadMotors
import rospy
import cv2



signal.signal(signal.SIGINT, signal.SIG_DFL)

if __name__ == '__main__':
    # Parameters
    rospy.init_node("followperson")
    topics = {'rgb': '/camera/rgb/image_raw',
              'depth': '/camera/depth_registered/image_raw'}

    network_model = 'ssdlite_mobilenet_v2_coco_2018_05_09.pb'
    

    # The camera does not need a dedicated thread, the callbacks have their owns.
    cam = ROSCam(topics)
    network = TrackingNetwork(network_model)
    network.setCamera(cam)
    display_imgs = False

    while not rospy.is_shutdown():
        if display_imgs:
            cv2.imshow("RGB", cam.rgb_img)
            cv2.imshow("depth", cam.depth_img)
            if cv2.waitKey(1) == 27:
                break
        # Make an inference on the current image
        network.predict()
        print "inference output", network.predictions, network.boxes, network.scores

    if display_imgs:
        cv2.destroyAllWindows()

    # t_network = ThreadNetwork(network)
    # t_network.start()



    # mom_path = cfg.getProperty('FollowPerson.Mom.ImagePath')

    # siamese_network = SiameseNetwork(siamese_model, mom_path)


    # app = QtWidgets.QApplication(sys.argv)
    # window = GUIClass()

    # if device_type.lower() == 'kobuki':
    #     depth_proxy = jdrc.getCameraClient('FollowPerson.Depth')
    #     depth = Camera(depth_proxy)
    #     network.setDepth(depth)
    #     t_depth = ThreadCamera(depth)
    #     t_depth.start()
    #     window.setCamera(cam, t_cam)
    #     window.setDepth(depth, t_depth)
    # else:
    #     window.setCamera(cam, t_cam)


    # motors = Motors(motors_proxy)
    # motors.setNetworks(network, siamese_network)
    # motors.setCamera(cam)
    # if device_type.lower() == 'kobuki':
    #     motors.setDepth(depth)
    # t_motors = ThreadMotors(motors)
    # window.setMotors(motors, t_motors)
    # t_motors.start()


    # window.setNetwork(network, t_network)
    # window.show()

    # # Threading GUI
    # t_gui = ThreadGUI(window)
    # t_gui.start()


    # print("")
    # print("Requested timers:")
    # print("    Camera: %d ms" % (t_cam.t_cycle))
    # print("    GUI: %d ms" % (t_gui.t_cycle))
    # print("    Network: %d ms" % (t_network.t_cycle))
    # print("    Motors: %d ms" % (t_motors.t_cycle))
    # print("")

    # sys.exit(app.exec_())
