#
# Created on Dec, 2019
#
# @author: naxvm
#
# This class creates a real-time camera to iterate over a
# labeled RGB-D sequence. Instead of delivering images sequentially,
# a framerate is defined, and the camera delivers images at that rate.
#
# This is done iterating over the sequence just on the required indices,
# depending on the time elapsed since the camera was started.


import numpy as np
import json
import cv2
import os
import utils
from datetime import datetime, timedelta

IMAGE_HEIGHT = 480
IMAGE_WIDTH = 640

__author__ = 'naxvm'

RGB_SUBDIR = 'RGB'
DEPTH_SUBDIR = 'Depth'
LABELS_SUBDIR = 'Labels'


class LabeledCam:

    def __init__(self, sequence_path, topics, is_bgr=False, rate=30.0):

        # Check the integrity of the files
        self.root_dir = sequence_path

        if not os.path.isdir(self.root_dir):
            raise FileNotFoundError(f"The path '{self.root_dir}' does not exist.")

        # Toggle for BGR->RGB conversion
        self.is_bgr = is_bgr

        self.init_time = None

        self.rgb_dir = os.path.join(self.root_dir, RGB_SUBDIR)
        if not os.path.isdir(self.rgb_dir):
            raise FileNotFoundError(f"{RGB_SUBDIR} directory missing inside the provided path '{self.root_dir}'.")

        self.depth_dir = os.path.join(self.root_dir, DEPTH_SUBDIR)
        if not os.path.isdir(self.rgb_dir):
            raise FileNotFoundError(f"{DEPTH_SUBDIR} directory missing inside the provided path '{self.root_dir}'.")

        self.labels_dir = os.path.join(self.root_dir, LABELS_SUBDIR)
        if not os.path.isdir(self.labels_dir):
            raise FileNotFoundError(f"{LABELS_SUBDIR} directory missing inside the provided path '{self.root_dir}'.")

        # Compute the number of images
        imgs = list(filter(lambda x: x.endswith('.jpg'), os.listdir(self.rgb_dir)))
        self.n_images = len(imgs)
        self.n_digits = len(str(self.n_images))

        self.rate = rate


    def init_cam(self):
        self.init_time = datetime.now()


    def __len__(self):
        return self.n_images


    def __getitem(self, idx):
        idxnum = str(idx).zfill(self.n_digits)
        image_path = os.path.join(self.rgb_dir, f"{idxnum}.jpg")
        depth_path = os.path.join(self.depth_dir, f"{idxnum}.npy")
        label_path = os.path.join(self.labels_dir, f"{idxnum}.json")

        # Check the existance for stopping the iteration when the images are consumed
        if not os.path.isfile(image_path):
            raise StopIteration(f'{idx} was not found. ({image_path})')

        image = cv2.imread(image_path)
        if not os.path.isfile(depth_path):
            depth = None
        else:
            depth = np.load(depth_path)

        instances = {}
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                label_data = json.load(f)
            instances = {}
            # Search for instances inside the label
            for inst in label_data['shapes']:
                label = inst['label']
                points = inst['points']
                if points[0][0] > points[1][0]:
                    coords = list(map(int, points[1] + points[0]))
                else:
                    coords = list(map(int, points[0] + points[1]))
                coords[0] = np.clip(coords[0], 0, image.shape[1])
                coords[1] = np.clip(coords[1], 0, image.shape[0])
                # coords = [int(point) for sub in inst['points'] for point in sub]
                coords = utils.corners2Corner(coords)
                instances[label] = coords

        return image, depth, instances



    def get_data(self):

        if self.init_time is None:
            print("Error: the camera has not been initialized!")
            return

        # Compute elapsed time
        el_since_start = datetime.now() - self.init_time

        # Convert to suitable index
        idx = int(el_since_start.total_seconds() * self.rate)
        # And get data
        image, depth, labels = self.__getitem(idx)
        return image, depth, labels

