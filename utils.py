#
# Created on Nov, 2019
# @author: naxvm
#

from datetime import datetime, timedelta
import yaml
from os import listdir, path
import numpy as np

META_KEY = '1 - Meta'
SUMMARY_KEY = '2 - Summary'
ITERS_KEY = '3 - Iterations'

FILENAME_FORMAT = '%Y%m%d %H%M%S.yml'

class BenchmarkWriter:
    ''' Class to save a benchmark result on a file. '''
    def __init__(self, logdir, network_model):
        self.logdir = logdir
        self.description = None
        self.network_model = network_model
        # self.check_last_commented()

    def check_last_commented(self):
        if not path.exists(self.logdir):
            print(f'Error: {self.path} does not exist.')
            return
        files = listdir(self.logdir)
        # Parse the filenames
        datetimes = list(map(lambda x: (datetime.strptime(x, FILENAME_FORMAT), x), files))
        datetimes.sort()
        # Read the most recent logging file
        latest_datetime, latest_file = datetimes[-1]
        with open(self.logdir + '/' + latest_file, 'r') as f:
            latest_log = yaml.safe_load(f)

        latest_descr = latest_log[META_KEY]['Description']
        
        if latest_descr != '':
            print(f'Latest description:\n{latest_descr}')
            self.description = input('Please enter the description for the new benchmark >> ')
            return True
        else:
            print(f'Error: the latest description on "{latest_file}" is empty')
            return False


    def write_log(self, elapsed_times, detections, optim_params=None):
        metrics = {}
        elapsed_ms = list(map(lambda x: x.seconds * 1000.0 + x.microseconds / 1000.0, elapsed_times))
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
