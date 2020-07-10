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
from faced import FaceDetector
from Perception.Net.utils import visualization_utils as vis_utils
from labelmesequence import LabelMeSequence
from time import sleep

VIDEO_PATH = 'own_videos/test_face'

if __name__ == '__main__':
    # Video containing the sequence and the ground truth labels
    video = LabelMeSequence(VIDEO_PATH)


    fdet = FaceDetector()
    frontal_face_cascade_classifier = cv2.CascadeClassifier('test2/haarcascade_frontalface_default.xml')

    results = []
    pbar = tqdm(total=len(video)-1)

    for image, labels in video:
        if image is not None:
            faces = list(map(utils.center2Corner, fdet.predict(image)))
            # image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            # faces = frontal_face_cascade_classifier.detectMultiScale(image, scaleFactor=1.1, minNeighbors=2)
            if len(faces) > 0 and len(labels) > 0:
                label = labels['0']
                JIs = [utils.jaccardIndex(label, face) for face in faces]
                results.append(max(JIs))
            else:
                results.append(np.nan)
        pbar.update(1)
    del pbar
    arr = np.array(results)
    fname = input('Enter name for the resulting CSV: ')
    np.savetxt(fname, arr, delimiter=';')
