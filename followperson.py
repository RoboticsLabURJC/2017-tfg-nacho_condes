#
# Created on Oct, 2017
#
# @author: naxvm
#
# It receives images from a live video and classify them into digits
# employing a convolutional neural network, based on TensorFlow Deep Learning middleware.
# It shows the live video and the results in a GUI.
#
# Based on @nuriaoyaga code:
# https://github.com/RoboticsURJC-students/2016-tfg-nuria-oyaga/blob/
#     master/numberclassifier.py
# and @dpascualhe's:
# https://github.com/RoboticsURJC-students/2016-tfg-david-pascual/blob/
#     master/digitclassifier.py
#
#

import sys
import signal

from PyQt5 import QtWidgets

from Camera.camera import Camera
from Camera.threadcamera import ThreadCamera
from GUI.gui import GUI
from GUI.threadgui import ThreadGUI
from Net.threadnetwork import ThreadNetwork

import config
import comm

signal.signal(signal.SIGINT, signal.SIG_DFL)

if __name__ == '__main__':


    try:
        cfg = config.load(sys.argv[1])
    except IndexError:
        raise SystemExit('Missing YML file. Usage: python2 followperson.py followperson.yml')

    jdrc = comm.init(cfg, 'FollowPerson')
    cam_proxy = jdrc.getCameraClient('FollowPerson.Camera')
    motors = jdrc.getPTMotorsClient('FollowPerson.PTMotors')


    net_prop = cfg.getProperty('Network')
    framework = net_prop['Framework']
    if framework.lower() == 'tensorflow':
        from Net.TensorFlow.network import TrackingNetwork
    elif framework.lower() == 'keras':
        sys.path.append('Net/Keras')
        from Net.Keras.network import TrackingNetwork
    else:
        raise SystemExit(('%s not supported! Supported frameworks: Keras, TensorFlow') % (framework))



    cam = Camera(cam_proxy)
    t_cam = ThreadCamera(cam)
    t_cam.start()

    network = TrackingNetwork(net_prop)
    network.setCamera(cam)
    network.setMotors(motors)
    t_network = ThreadNetwork(network)
    t_network.start()

    app = QtWidgets.QApplication(sys.argv)
    window = GUI()
    window.setCamera(cam, t_cam)
    window.setNetwork(network, t_network)
    window.show()

    # Threading GUI
    t_gui = ThreadGUI(window)
    t_gui.start()


    print("")
    print("Requested timers:")
    print("    Camera: %d ms" % (t_cam.t_cycle))
    print("    GUI: %d ms" % (t_gui.t_cycle))
    print("    Network: %d ms" % (t_network.t_cycle))
    print("")

    sys.exit(app.exec_())
