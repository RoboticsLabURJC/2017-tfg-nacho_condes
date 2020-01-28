import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import seaborn as sns
import yaml
from utils import TO_MS
from os import listdir, path, makedirs


sns.set()

FILENAME_FORMAT = '%Y%m%d %H%M%S.yml'
# TO_MS = np.vectorize(lambda x: x.seconds * 1000.0 + x.microseconds / 1000.0) # Auxiliary vectorized function

class Benchmarker:
    ''' Writer for a benchmark using a certain configuration. '''
    def __init__(self, logdir):
        self.logdir = logdir
        self.description = None

    def write_benchmark(self, times_list, rosbag_file,
                        pdet_model, fenc_model,
                        t_pers_det, t_face_det, t_face_enc, ttfi,
                        display_images, write_iters=True, dirname=None):
        ''' Write the metrics to the output file. '''

        benchmark = {}
        summary = {}
        summary['1.- Configuration'] = {
            '1.- Networks': {
                '1.- PersonDetectionModel': pdet_model,
                # '2.- FaceDetectionModel':   fdet_model,
                '2.- FaceEncodingModel':    fenc_model,
            },
            '2.- RosbagFile': rosbag_file,
            '3.- DisplayImages': display_images,
        }
        summary['2.- LoadTimes'] = {
            '1.- PersonDetectionNetworkLoad': float(TO_MS(t_pers_det)),
            '2.- FaceDetectionNetworkLoad':   float(TO_MS(t_face_det)),
            '3.- FaceEncodingNetworkLoad':    float(TO_MS(t_face_enc)),
            '4.- TTFI':                       float(TO_MS(ttfi)),
        }

        # Process the measured times
        times_raw = np.array(times_list)
        # Split dropping the first (slower) inference
        iters_raw = times_raw[1:, 0]
        total_iters = TO_MS(iters_raw)

        pdets_raw = np.array(list(times_raw[1:, 1]))
        total_pdets = pdets_raw.copy()
        total_pdets[:, 0] = TO_MS(total_pdets[:, 0])

        fdets_raw = np.array(list(times_raw[1:, 2]))
        total_fdets = fdets_raw.copy()
        total_fdets[:, 0] = TO_MS(fdets_raw[:, 0])

        fencs_raw = np.array(list(times_raw[1:, 3]))
        total_fencs = fencs_raw.copy()
        total_fencs[:, 0] = TO_MS(total_fencs[:, 0])
        total_fencs_flt = total_fencs[total_fencs[:, 1] > 0]  # Just times belonging to a face filtering
        # total_fencs_flt = list(filter(lambda x: x[1] > 0, total_fencs))

        if display_images:
            disps_raw = times_raw[1:, 4]
            total_disps = TO_MS(disps_raw)

        summary['3.- Stats'] = {
            '1.- PersonDetection': {
                '1.- Mean': float(total_pdets.mean()),
                '2.- Std':  float(total_pdets.std()),
            },
            '2.- FaceDetection': {
                '1.- Mean': float(total_fdets.mean()),
                '2.- Std':  float(total_fdets.std()),
            },
            '3.- FaceEncoding': {
                '1.- Mean': float(total_fencs_flt.mean()),
                '2.- Std':  float(total_fencs_flt.std()),
            }
        }
        benchmark['1.- Summary'] = summary

        if write_iters:
            iterations = []
            for it_time, pdet, fdet, fenc in zip(total_iters, total_pdets, total_fdets, total_fencs):
                iteration = {}
                # Persons detection
                persons_detection = {}
                n_persons = len(pdet)
                persons_detection['1.- NumDetections'] = int(n_persons)
                if n_persons == 0:
                    continue
                persons_detection['2.- TotalTime'] = float(pdet.sum())
                # Faces detection
                faces_detection = {}
                n_faces = fdet[1]
                faces_detection['1.- NumDetections'] = int(n_faces)
                if n_faces == 0:
                    continue
                faces_detection['2.- TotalTime'] = float(fdet.sum())
                # Faces encoding
                faces_encoding = {}
                n_faces = fenc[1]
                faces_encoding['1.- NumDetections'] = int(n_faces)
                if n_faces == 0:
                    continue
                faces_encoding['2.- TotalTime'] = float(fenc.sum())

                iteration = {
                    '1.- PersonsDetection':   persons_detection,
                    '2.- FacesDetection':     faces_detection,
                    '3.- FacesEncoding':      faces_encoding,
                    '4.- TotalIterationTime': float(it_time),
                }
                iterations.append(iteration)

            benchmark['2.- Iterations'] = iterations

        if dirname is None:
            dirname = path.join(self.logdir, datetime.now().strftime(FILENAME_FORMAT))

        dirname = path.join(self.logdir, dirname)
        if not path.exists(dirname):
            makedirs(dirname)
        benchmark_name = path.join(dirname, 'benchmark.yml')

        # Dump
        with open(benchmark_name, 'w') as f:
            yaml.dump(benchmark, f)

        print(f'Saved on {benchmark_name}')
        # Graphs
        #   Total iteration time
        fig, ax = plt.subplots()
        ax.plot(total_iters)
        ax.set_title('Total iteration time')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Time (ms)')
        figname = path.join(dirname, 'iterations.png')
        fig.savefig(figname)

        #   Person detection time
        fig, ax = plt.subplots()
        ax.plot(total_pdets[:, 0])
        ax.set_title('Person detection time')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Time (ms)')
        figname = path.join(dirname, 'person_detections.png')
        fig.savefig(figname)

        #   Face detection time
        fig, ax = plt.subplots()
        ax.plot(total_fdets[:, 0])
        ax.set_title('Face detection time')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Time (ms)')
        figname = path.join(dirname, 'face_detections.png')
        fig.savefig(figname)

        #   Face encoding time
        fig, ax = plt.subplots()
        ax.plot(total_fencs[:, 0])
        ax.set_title('Face encoding time')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Time (ms)')
        figname = path.join(dirname, 'face_encoding.png')
        fig.savefig(figname)
