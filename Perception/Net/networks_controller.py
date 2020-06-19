#
# Created on Mar. 2020
#

__author__ = '@naxvm'

import threading
import time
from datetime import datetime

import utils
from cprint import cprint
from utils import crop_face
from imageio import imread
from faced import FaceDetector
from Perception.Net.facenet import FaceNet
from Perception.Net.detection_network import DetectionNetwork


class NetworksController(threading.Thread):

    def __init__(self, nets_cfg, ref_img_path, benchmark=False):
        """ Threading class for running neural inferences on a sequential way. """

        super(NetworksController, self).__init__()

        # Arguments for the networks
        self.nets_cfg = nets_cfg
        self.ref_img_path = ref_img_path

        # Placeholders
        self.pdet_network = None
        self.fdet_network = None
        self.fenc_network = None

        self.image = []
        self.depth = []
        self.frame_counter = 0

        self.persons = []
        self.faces = []
        self.similarities = []

        # Timing purposes
        self.last_elapsed = 0
        self.is_activated = False

        # Benchmarking purposes
        self.benchmark = benchmark
        self.total_times = {}
        self.t_pers_det = None
        self.t_face_det = None
        self.t_face_enc = None
        self.ttfi = None

        # self.cam = None
        self.tracker = None

    def createPersonDetector(self):
        """Instantiate the "person" detection network."""
        start = datetime.now()
        input_shape = (self.nets_cfg['DetectionHeight'], self.nets_cfg['DetectionWidth'], 3)
        pdet_network = DetectionNetwork(self.nets_cfg['Arch'], input_shape, self.nets_cfg['DetectionModel'])
        elapsed = datetime.now() - start
        # Assign the attributes
        self.pdet_network = pdet_network
        self.t_pers_det = elapsed

    def createFaceDetector(self):
        '''Instantiate the face detection network.'''
        start = datetime.now()
        fdet_network = FaceDetector()
        elapsed = datetime.now() - start
        # Assign the attributes
        self.fdet_network = fdet_network
        self.t_face_det = elapsed

    def createFaceEncoder(self):
        '''Instantiate the face encoding network.'''
        start = datetime.now()
        fenc_network = FaceNet(self.nets_cfg['FaceEncoderModel'])
        elapsed = datetime.now() - start
        # Assign the attributes
        self.fenc_network = fenc_network
        self.t_face_enc = elapsed


    # def setCam(self, camera):
    #     """Set the camera object and get the first RGB-D pair."""
    #     self.cam = camera
    #     self.image, self.depth = self.cam.getImages()
    #     self.frame_counter += 1

    def setTracker(self, tracker):
        """Set the tracker (CPU thread to be updated with the
        latest inferences."""
        self.tracker = tracker
        self.image = self.tracker.image
        self.depth = self.tracker.depth


    def run(self):
        """Main method of the thread."""

        # Create the networks
        zero_time = datetime.now()
        self.createPersonDetector()
        self.createFaceDetector()
        self.createFaceEncoder()

        # Set the reference face
        ref_img = imread(self.ref_img_path)
        ref_box = self.fdet_network.predict(ref_img)
        ref_face = crop_face(ref_img, ref_box)
        self.fenc_network.setReferenceFace(ref_face)

        self.ttfi = datetime.now() - zero_time
        # Indicate we are ready to go
        self.is_activated = True
        self.tracker.is_activated = True
        while self.is_activated:
            print('net!!!')
            iter_info = []
            # Fetch the images
            try:
                # self.image, self.depth = self.cam.getImages()
                # We get it from the tracker, in order not to consume the
                # iterator if the images come from a ROSBag
                self.is_activated = self.tracker.is_activated
                self.image = self.tracker.image
                self.depth = self.tracker.depth
                self.frame_counter += 1
            except StopIteration:
                self.is_activated = False
                break
            if self.benchmark:
                step_time = datetime.now()
                iter_start = step_time



            ### Person detection ###
            print('la shape')
            print(self.image.shape, self.image.dtype, self.image.min(), self.image.max())
            self.persons, elapsed = self.pdet_network.predict(self.image)
            if self.benchmark:
                iter_info.append([elapsed, len(self.persons)])
                step_time = datetime.now()



            ### Face detection and cropping ###
            face_detections = self.fdet_network.predict(self.image)
            if self.benchmark:
                elapsed = datetime.now() - step_time
                iter_info.append([elapsed, len(face_detections) if isinstance(face_detections, list) else 1])

            # Just confident faces
            self.faces = list(filter(lambda f: f[-1] > 0.9, face_detections))
            faces_cropped = [utils.crop_face(self.image, fdet) for fdet in self.faces]
            if self.benchmark: step_time = datetime.now()



            ### Face similarities ###
            self.similarities = self.fenc_network.distancesToRef(faces_cropped)
            if self.benchmark:
                elapsed = datetime.now() - step_time
                iter_info.append([elapsed, len(self.similarities)])

            # Make the tracking thread to update the persons
            self.tracker.updateWithDetections(self.persons, self.faces, self.similarities)
            # self.tracker.faces = self.faces
            # self.tracker.similarities = self.similarities

            # Finishing the loop
            if self.benchmark:
                iter_elapsed = datetime.now() - iter_start
                self.last_elapsed = iter_elapsed
                iter_info.append(iter_elapsed)
                self.total_times[self.frame_counter] = iter_info

                # cprint.info(f'\r[INFERENCES] Elapsed: {iter_elapsed}\t{1e6/iter_elapsed.microseconds:.2f} fps', end='', flush=True)


    def close_all(self):
        '''Function to stop the inferences.'''
        self.is_activated = False
        # Finish current inferences
        time.sleep(1)

        self.pdet_network.sess.close()
        self.fdet_network.sess.close()
        self.fenc_network.sess.close()
        print('All the sessions were closed.')
