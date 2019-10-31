#
# Created on May, 2018
#
# @author: naxvm
#
# Based on dl-objectdetector
# https://github.com/jderobot/dl-objectdetector

import sys
import signal
import rospy
import cv2
import numpy as np
from datetime import datetime, timedelta
from faced import FaceDetector

from Camera.ROSCam import ROSCam
from Net.network import DetectionNetwork
from Net.siamese_network import FaceTrackingNetwork
from Motors.Kobuki.motors import Motors


if __name__ == '__main__':
    rospy.init_node("followperson")
    # Parameters
    # TODO: read from YML config file
    topics = {'rgb':      '/camera/rgb/image_raw',
              'depth':    '/camera/depth_registered/image_raw',
              'velocity': '/mobile_base/commands/velocity',}

    network_model = 'ssdlite_mobilenet_v2_coco_2018_05_09.pb'
    siamese_model = 'facenet_model.pb'
    mom_img = 'mom_img/mom.jpg'


    # The camera does not need a dedicated thread, the callbacks have their owns.
    cam = ROSCam(topics)
    # Neural network TF instances
    network = DetectionNetwork(network_model)
    face_detector = FaceDetector()
    siamese_network = FaceTrackingNetwork(siamese_model, mom_img, face_detector)
    # Motors instance
    motors = Motors(topics['velocity'])
    motors.setNetworks(network, siamese_network)
    # network.setCamera(cam)
    display_imgs = True
    MAX_ITER = 100

    iteration = 0
    elapsed_times = []

    def shtdn_hook():
        network.sess.close()
        face_detector.sess.close()
        siamese_network.sess.close()
        print("Cleaning and exiting...")
    # Register shutdown hook
    rospy.on_shutdown(shtdn_hook)

    while not rospy.is_shutdown():
        # Make an inference on the current image
        image = cam.rgb_img
        depth = cam.depth_img
        start_time = datetime.now()
        persons, scores = network.predict(image)
        p1_time = datetime.now()
        elapsed_1 = p1_time - start_time

        faces = face_detector.predict(image)
        elapsed_2 = datetime.now() - p1_time
        print("faces\t", faces)
        print("persons\t", persons)
        elapsed_3 = motors.move(image, depth, persons, scores, faces)


    # motors.setNetworks(network, siamese_network)
    # motors.setCamera(cam)
    # if device_type.lower() == 'kobuki':
    #     motors.setDepth(depth)

        # for face in faces:
        #     for person in boxes:
        #         # We check the face center belongs to a person
        #         horz = person[0] <= face[0] <= person[2]
        #         vert = person[1] <= face[1] <= person[3]
        #         print horz, vert





        elapsed_times.append([elapsed_1, elapsed_2, elapsed_3])
        iteration += 1
        print(iteration)
        print('-----#~~~~~~~~----------')
        if iteration == MAX_ITER:
            rospy.signal_shutdown("Finished!!")

        # print "elapsed {} ms. Framerate: {} fps".format(elapsed.microseconds/1000.0, 1e6/elapsed.microseconds)
        # print "inference output", network.predictions, network.boxes, network.scores
        # print "faces:", faces
        # print ""
        # Draw every detected person
        # for idx, person in enumerate(network.boxes):
        #     [xmin, ymin, xmax, ymax] = person
        #     cv2.rectangle(img_cp, (xmin, ymax), (xmax, ymin), (0,255,0), 5)
        # if display_imgs:
        #     cv2.imshow("RGB", img_cp)
        #     cv2.imshow("depth", cam.depth_img)
        #     if cv2.waitKey(1) == 27:
        #         break
    print("\n\n\nElapsed times:")
    t1s = []
    t2s = []
    t3s = []
    for t1, t2, t3 in elapsed_times:
        t1s.append(t1.microseconds / 1000.0)
        t2s.append(t2.microseconds / 1000.0)
        if type(t3) == timedelta:
            t3s.append(t3.microseconds / 1000.0)
    print("Means\n-----\n%.3f\t%.3f\t%.3f" %(np.mean(t1s), np.mean(t2s), np.mean(t3s)))
    print("Stds\n----\n%.3f\t%.3f\t%.3f" %(np.std(t1s), np.std(t2s), np.std(t3s)))
    if display_imgs:
        cv2.destroyAllWindows()













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
