#
# Created on Mar. 2020
#

__author__ = '@naxvm'

import threading
import time
from datetime import datetime

from cprint import cprint

import utils


class NetworksController(threading.Thread):

    def __init__(self, pdet_network, fdet_network, fenc_network, benchmark=False):
        ''' Threading class for running neural inferences on a sequential way. '''

        super(NetworksController, self).__init__()

        # Set the attributes
        self.pdet_network = pdet_network
        self.fdet_network = fdet_network
        self.fenc_network = fenc_network

        self.image = []
        self.depth = []

        self.persons = []
        self.faces_flt = []
        self.similarities = []

        # self.framerate = 0
        self.is_activated = True
        # self.t_cycle = 50  # ms

        # Benchmarking purposes
        self.benchmark = benchmark
        self.total_times = []

    def setCamera(self, camera):
        self.cam = camera
        self.image, self.depth = self.cam.getImages()

    def run(self):
        ''' Updates the thread. '''
        while self.is_activated:
            iter_info = []
            # Fetch the images
            try:
                self.image, self.depth = self.cam.getImages()
            except StopIteration:
                self.is_activated = False
                break
            if self.benchmark:
                step_time = datetime.now()
                iter_start = step_time



            # Person detection
            self.persons, elapsed = self.pdet_network.predict(self.image)
            if self.benchmark:
                iter_info.append([elapsed, len(self.persons)])
                step_time = datetime.now()



            # Face detection and cropping
            face_detections = self.fdet_network.predict(self.image)
            if self.benchmark:
                elapsed = datetime.now() - step_time
                iter_info.append([elapsed, len(face_detections) if isinstance(face_detections, list) else 1])

            # Just confident faces
            self.faces_flt = list(filter(lambda f: f[-1] > 0.9, face_detections))
            # TODO: filter the faces by position?
            faces_cropped = [utils.crop_face(self.image, fdet) for fdet in self.faces_flt]
            if self.benchmark: step_time = datetime.now()



            # Face similarities
            self.similarities = self.fenc_network.distancesToRef(faces_cropped)
            if self.benchmark:
                elapsed = datetime.now() - step_time
                iter_info.append([elapsed, len(similarities) if isinstance(self.similarities, list) else 1])



            # Finishing the loop
            if self.benchmark:
                iter_elapsed = datetime.now() - iter_start
                iter_info.append(iter_elapsed)
                self.total_times.append(iter_info)
            print(f'\rElapsed: {iter_elapsed}\t{1e6/iter_elapsed.microseconds:.2f} fps', end='', flush=True)


    def close_all(self):
        '''Function to stop the inferences.'''
        self.is_activated = False
        # Finish current inferences
        time.sleep(1)

        self.pdet_network.sess.close()
        self.fdet_network.sess.close()
        self.fenc_network.sess.close()
        print('All the sessions were closed.')
