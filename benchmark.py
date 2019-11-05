#
# Created on Oct, 2019
#
# @author: naxvm
#
# Tester for benchmarking the neural implementations performance.

import rospy
import cv2
import numpy as np
from datetime import datetime, timedelta
from faced import FaceDetector
import sys
import yaml

from Camera.ROSCam import ROSCam
from Net.network import DetectionNetwork
from utils import BenchmarkWriter
# from Net.siamese_network import FaceTrackingNetwork
# from Motors.motors import Motors

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Error: missing configuration file!")
        exit()
    # Parse the config file
    with open(sys.argv[1], 'r') as f:
        try:
            config = yaml.safe_load(f)['DetectionTester']
        except yaml.YAMLError as exc:
            print(exc)
            exit()

    IMG_TOPICS = config['ImageTopics']
    NETWORK_MODEL = config['NetworkModel']
    LOGDIR = config['LogDir']

    bm_wrt = BenchmarkWriter(LOGDIR, NETWORK_MODEL)  # Benchmark writer
    if not bm_wrt.check_last_commented():
        exit()
    cam = ROSCam(IMG_TOPICS)                         # Camera subscribers
    network = DetectionNetwork(NETWORK_MODEL)        # Network instance

    rospy.init_node(config['NodeName'])

    MAX_ITER = 200

    def shtdn_hook():
        network.sess.close()
        print('Cleaning and exiting...')
    # Register shutdown hook
    rospy.on_shutdown(shtdn_hook)

    iteration = 0
    elapsed_times = []
    detections = []
    while not rospy.is_shutdown():
        # Make an inference on the current image
        image = cam.rgb_img
        start_time = datetime.now()
        persons, scores = network.predict(image)
        elapsed = datetime.now() - start_time
        detections.append([persons, scores])
        elapsed_times.append(elapsed)
        iteration += 1
        print(f'{iteration}/{MAX_ITER}')

        if iteration == MAX_ITER:
            bm_wrt.write_log(elapsed_times, detections)
            rospy.signal_shutdown('Finished!!')
