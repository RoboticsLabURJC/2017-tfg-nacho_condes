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
# from Camera.threadcamera import ThreadCamera
from GUI.threadgui import ThreadGUI
from Net.threadnetwork import ThreadNetwork
from Motors.threadmotors import ThreadMotors
import rospy



signal.signal(signal.SIGINT, signal.SIG_DFL)

if __name__ == '__main__':

    rospy.init_node("followperson")
    topics = {'rgb': '/camera/rgb/image_raw',
              'depth': '/camera/depth_registered/image_raw'}

    # try:
    #     cfg = config.load(sys.argv[1])
    # except IndexError:
    #     raise SystemExit('Missing YML file. Usage: python2 followperson.py [your_config_file].yml')

    # jdrc = comm.init(cfg, 'FollowPerson')
    # cam_proxy = jdrc.getCameraClient('FollowPerson.Camera')


    # Network (TensorFlow/Keras) parsing:
    # net_prop = cfg.getProperty('FollowPerson.Network')
    # framework = net_prop['Framework']
    # if framework.lower() == 'tensorflow':
    #     from Net.TensorFlow.network import TrackingNetwork
    #     # Parse and import the siamese network for face identification
    #     siamese_model = net_prop['SiameseModel']
    #     from Net.TensorFlow.siamese_network import SiameseNetwork
    # elif framework.lower() == 'keras':
    #     from Net.Keras.network import TrackingNetwork
    # else:
    #     raise SystemExit(('%s not supported! Supported frameworks: Keras, TensorFlow') % (framework))

    # # Device (PTZ/Kobuki) parsing:
    # device_type = cfg.getProperty('FollowPerson.Device')
    # if device_type.lower() == 'kobuki':
    #     # GUI version with depth image
    #     from GUI.gui import DepthGUI as GUIClass
    #     # Turtlebot motors
    #     from Motors.Kobuki.motors import Motors
    #     motors_proxy = jdrc.getMotorsClient('FollowPerson.Motors')
    #     # PT motors for EVI camera
    # elif device_type.lower() == 'ptz':
    #     from GUI.gui import GUI as GUIClass
    #     from Motors.PTZ.motors import Motors
    #     motors_proxy = jdrc.getPTMotorsClient('FollowPerson.PTMotors')
    # else:
    #     raise SystemExit(('%s not supported! Supported devices: Kobuki, PTZ') % (device_type))

    import cv2

    cam = ROSCam(topics)
    while not rospy.is_shutdown():
        cv2.imshow("RGB", cam.rgb_img)
        cv2.imshow("depth", cam.depth_img)
        if cv2.waitKey(1) == 27:
            break
    cv2.destroyAllWindows()

    # network = TrackingNetwork(net_prop)
    # network.setCamera(cam)
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
