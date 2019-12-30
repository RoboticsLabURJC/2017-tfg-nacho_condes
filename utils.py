#
# Created on Nov, 2019
# @author: naxvm
#

from datetime import datetime, timedelta
import yaml
from os import listdir, path
import numpy as np

FILENAME_FORMAT = '%Y%m%d %H%M%S.yml'

TO_MS = np.vectorize(lambda x: x.seconds * 1000.0 + x.microseconds / 1000.0) # Auxiliary vectorized function


def log_benchmark(logdir, parameters, person_detections=None, person_times=None,
                                      face_detections=None, face_times=None):
        if parameters.check_persons:
            person_avg_ms = np.vectorize(TO_MS)


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



class BenchmarkWriter:
    ''' Class to save a benchmark result on a file. '''
    def __init__(self, logdir, network_model):
        self.logdir = logdir
        self.description = None
        self.network_model = network_model

    def write_log(self, elapsed_times, detections, optim_params=None, write_iters=True):
        metrics = {}
        to_ms = lambda x: x.seconds * 1000.0 + x.microseconds / 1000.0 # Auxiliary function
        elapsed_ms = np.vectorize(to_ms)(elapsed_times)
        elapsed_mean = float(np.mean(elapsed_ms))
        elapsed_std = float(np.std(elapsed_ms))
        avg_fps = float(1000.0 / elapsed_mean)
        summary = {
            'AverageInferenceTime': elapsed_mean,
            'StdInferenceTime': elapsed_std,
            'AverageFPS': avg_fps,
        }
        if optim_params is not None:
            metrics[META_KEY] = {
                'OriginalModel': optim_params['model_name'],
                'PrecisionMode': optim_params['prec'],
                'MinimumSegmentSize': optim_params['mss'],
                'MaximumCachedEngines': optim_params['mce'],
            }
        else:
            metrics[META_KEY] = {
                'Datetime': datetime.now().strftime('%d-%m-%Y %H:%M:%S'),
                'Description': self.description,
                'NetworkModel': self.network_model,
            }
        metrics[SUMMARY_KEY] = summary
        metrics[ITERS_KEY] = {}
        if write_iters:
            for (idx, elp), dets in zip(enumerate(elapsed_ms), detections):
                det = {
                    'ElapsedMs':     elp,
                    'NumDetections': len(dets[0])
                }
                metrics['3 - Iterations'][idx + 1] = det

            filename = datetime.now().strftime(self.logdir + '/' + FILENAME_FORMAT)
            with open(filename, 'w') as f:
                yaml.dump(metrics, f)

            print(f'Logfile saved on {filename}')
