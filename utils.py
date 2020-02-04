#
# Created on Nov, 2019
# @author: naxvm
#

from datetime import datetime, timedelta
import yaml
from os import listdir, path
import numpy as np

FILENAME_FORMAT = '%Y%m%d %H%M%S.yml'

# TO_MS = np.vectorize(lambda x: x.seconds * 1000.0 + x.microseconds / 1000.0) # Auxiliary vectorized function


def crop_face(image, det):
    ''' Crop the detected face, using the faced detection outputs. '''
    cx, cy, w, h, prob = np.squeeze(det).astype(int)

    # Filter as the borders might fall outside the image
    im_h, im_w = image.shape[:2]

    y_up =    max(0,    cy - h//2)
    y_down =  min(im_h, cy + h//2)
    x_left =  max(0,    cx - w//2)
    x_right = min(im_w, cx + w//2)

    face_crop = image[y_up:y_down, x_left:x_right]

    return face_crop
