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

def distanceBetweenBoxes(bb1, bb2):
    '''Compute the center of both boxes and return the distance betwen them.'''
    bb1 = np.array(bb1, dtype=np.int32)
    bb2 = np.array(bb2, dtype=np.int32)
    center1 = np.divide([bb1[3] + bb1[1], bb1[2] + bb1[0]], 2)
    center2 = np.divide([bb2[3] + bb2[1], bb2[2] + bb2[0]], 2)

    distance = np.linalg.norm(center1 - center2)
    return distance


def bb1inbb2(bb1, bb2):
    '''Check whether a bounding box is contained inside another one.
    Both in corner-width-height format.'''
    c1x = bb1[0] >= bb2[0]
    c1y = bb1[1] >= bb2[1]
    c2x = bb1[0] + bb1[2] <= bb2[0] + bb2[2]
    c2y = bb1[1] + bb1[3] <= bb2[1] + bb2[3]

    return c1x and c1y and c2x and c2y

def computeWError(coords, im_width):
    '''Return the error (in px) between the center of the image and the center
    of the tracked person.'''
    person_center = coords[0] + coords[2]/2

    w_error = im_width/2 - person_center
    return w_error

def computeXError(coords, depth):
    '''Compute the depth error, sampling the depth image inside the detected box.'''

    # Crop the depth from the person with inner padding of 10%
    coords = np.array(coords, dtype=np.int)
    bb_depth = depth[coords[1]:coords[1]+coords[3], coords[0]:coords[0]+coords[2]]
    ph, pw = bb_depth.shape
    cropped_depth = bb_depth[ph//10:-ph//10, pw//10:-pw//10]

    # Create a 10x10 mesh to sample the depth values
    vg = np.linspace(0, cropped_depth.shape[0]-1, num=10, dtype=np.int)
    hg = np.linspace(0, cropped_depth.shape[1]-1, num=10, dtype=np.int)
    grid = tuple(np.meshgrid(vg, hg))

    # Sample the values and compute the median depth
    sampled_depths = cropped_depth[grid].ravel()
    median = np.nanmedian(sampled_depths)
    if np.isnan(median):
        # The person is too close to estimate the distance
        # We report 0 distance
        return 0.0
    return median

