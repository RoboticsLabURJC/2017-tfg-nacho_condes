#
# Created on June, 2020
#
#
__author__ = '@naxvm'

import argparse
import os

from tqdm import tqdm
import numpy as np
import yaml
from benchmarkers import TO_MS
import cv2
# import rospy
import utils
# from Perception.Camera.ROSCam import IMAGE_HEIGHT, IMAGE_WIDTH, ROSCam
from Perception.Net.detection_network import DetectionNetwork
from Perception.Net.utils import visualization_utils as vis_utils
from labelmesequence import LabelMeSequence
from time import sleep

SSD_PATH = 'Optimization/dl_models/ssd_mobilenet_v1_0.75_depth_coco/frozen_inference_graph.pb'
YOLO_PATH = 'Optimization/dl_models/ssd_mobilenet_v1_0.75_depth_coco/optimizations/FP16_50_1.pb'
VIDEO_PATH = 'own_videos/test'
PICKLE_NAME = 'test3/ground_truth.pkl'

if __name__ == '__main__':
    # Video containing the sequence and the ground truth labels
    video = LabelMeSequence(VIDEO_PATH)

    # net = DetectionNetwork('ssd', (300, 300, 3), frozen_graph=SSD_PATH)
    net = DetectionNetwork('yolov3tiny', (416, 416, 3), frozen_graph=YOLO_PATH)

    results = []
    pbar = tqdm(total=len(video)-1)

    for image, labels in video:
        if image is not None:
            persons, elapsed = net.predict(image)
            if len(persons) > 0 and len(labels) > 0:
                label = labels['0']
                JIs = [utils.jaccardIndex(label, person) for person in persons]
                results.append([max(JIs), TO_MS(elapsed)])
            else:
                results.append([np.nan, TO_MS(elapsed)])
        pbar.update(1)
    del pbar
    arr = np.array(results)
    fname = input('Enter name for the resulting CSV: ')
    np.savetxt(fname, arr, delimiter=';')
