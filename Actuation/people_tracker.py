import cv2
import threading
import time
from datetime import datetime
from cprint import cprint
import utils
from Actuation.tracking_classes import *
import numpy as np
np.set_printoptions(precision=2)
FEATURE_PARAMS = dict(maxCorners=120,
                      qualityLevel=0.3,
                      minDistance=7,
                      blockSize=7)

LK_PARAMS = dict(winSize=(15, 15),
                 maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

PERIOD = 1/30   # time elapsed between frames on a 30 fps sensor


class PeopleTracker(threading.Thread):
    """This class creates a thread responsible of continuously tracking the detected persons
     in the image."""

    def __init__(self, patience, ref_sim_thr, same_person_thr, debug=False):
        super(PeopleTracker, self).__init__()
        self.name = 'PeopleTrackerThread'
        self.daemon = True
        # Placeholders
        self.persons = []
        self.candidates = []
        self.tracked_counter = 0
        self.keypoints = []
        self.image = []
        self.gray_image = []
        self.depth = []
        # Parameters
        self.same_person_thr = same_person_thr
        self.ref_sim_thr = ref_sim_thr
        self.patience = patience
        self.cam = None
        self.frame_counter = 0
        # self.faces = []
        # self.similarities = []

        self.is_activated = False
        self.lock = threading.Lock()
        self.debug = debug

    def setCam(self, cam):
        self.cam = cam
        self.image, self.depth = self.cam.getImages()
        self.frame_counter += 1


    def setPrior(self):
        """Set the first image on the tracker."""
        self.gray_image = cv2.cvtColor(self.image, cv2.COLOR_RGB2GRAY)
        keypoints = cv2.goodFeaturesToTrack(self.gray_image, **FEATURE_PARAMS)
        self.keypoints = keypoints.reshape(-1, 2)
        # self.keypoints = keypoints.squeeze()


    def getImages(self):
        """Serve the latest available images from the Camera."""
        return self.image, self.depth


    def stepAll(self):
        """Propagate the candidate/tracked persons using the latest image."""
        new_image = cv2.cvtColor(self.image, cv2.COLOR_RGB2GRAY)
        new_kps, status, errors = cv2.calcOpticalFlowPyrLK(self.gray_image, new_image, self.keypoints, None,
                                                           **LK_PARAMS)
        # Retain only found keypoints
        found_idx = status.squeeze() == 1
        old_found = self.keypoints[found_idx]
        new_found = new_kps[found_idx]

        # And compute the individual displacements for each person
        for candidate in self.candidates:
            candidate.step(old_found, new_found)
        for person in self.persons:
            person.step(old_found, new_found)

        # Update the reference frame and keypoints
        self.gray_image = new_image
        self.keypoints = new_kps

    def updateWithDetections(self, boxes, faces, similarities):
        """Reassign the person to the most suitable bounding box."""
        print(self.persons)
        for box in boxes:
            pers_distances = np.array(list(map(lambda x: utils.distanceBetweenBoxes(box, x.coords), self.persons)))
            cand_distances = np.array(list(map(lambda x: utils.distanceBetweenBoxes(box, x.coords), self.candidates)))

            # Assign to the nearest person, or create one if required
            near_pers = np.where(pers_distances <= self.same_person_thr)[0]
            near_cand = np.where(cand_distances <= self.same_person_thr)[0]

            if len(near_pers) > 0 and len(near_cand) == 0:
                # Closest close person
                lowest_idx = np.argmin(pers_distances[near_pers])
                if isinstance(lowest_idx, np.ndarray):
                    lowest_idx = lowest_idx[0]
                self.persons[lowest_idx].coords = box
                # The person is still found
                self.persons[lowest_idx].counter = self.patience
            elif len(near_cand) > 0 and len(near_pers) == 0:
                # Closest close candidate
                lowest_idx = np.argmin(cand_distances[near_cand])
                if isinstance(lowest_idx, np.ndarray):
                    lowest_idx = lowest_idx[0]
                self.candidates[lowest_idx].coords = box
                # The candidate is still found
                self.candidates[lowest_idx].counter += 2
            elif len(near_cand) > 0 and len(near_pers) > 0:
                # Nearby candidate and persons. Pick the closest
                min_dist_cand = min(cand_distances[near_cand])
                min_dist_pers = min(pers_distances[near_pers])
                if min_dist_cand < min_dist_pers:
                    lowest_idx = np.argmin(cand_distances[near_cand])
                    if isinstance(lowest_idx, np.ndarray):
                        lowest_idx = lowest_idx[0]
                    self.candidates[lowest_idx].coords = box
                    self.candidates[lowest_idx].counter += 2
                else:
                    lowest_idx = np.argmin(pers_distances[near_pers])
                    if isinstance(lowest_idx, np.ndarray):
                        lowest_idx = lowest_idx[0]
                    self.persons[lowest_idx].coords = box
                    self.persons[lowest_idx].counter = self.patience
            else:
                # This detection can't be assigned to anyone. We create a new candidate
                candidate = Person(box)
                self.candidates.append(candidate)
        # And refresh the present persons with the new information
        self.handleFaces(faces, similarities)
        self.checkRef()
        self.refresh()


    def refresh(self):
        """Update the stored persons."""
        new_persons = []
        new_candidates = []
        # Candidates:
        for cand in self.candidates:
            if cand.counter >= 0:
                # Survive
                if cand.counter >= self.patience:
                    # The candidate will be a tracked person
                    cand.counter = self.patience
                    new_persons.append(cand)
                else:
                    # Still candidate
                    cand.counter -= 1
                    new_candidates.append(cand)
        self.candidates = new_candidates

        # Persons:
        for person in self.persons:
            if person.counter >= 0:
                # Survive
                person.counter -= 1
                new_persons.append(person)
        self.persons = new_persons

    def handleFaces(self, faces, similarities):
        """Check if a detected face belongs (spatially) to a person, and track it. Discard it otherwise."""
        for face, sim in zip(faces, similarities):
            face_std = utils.center2Corner(face)
            for person in self.persons:
                if bb1inbb2(face_std, person.coords):
                    # The face is inside this person's box.
                    person.setFace(face, sim)


        for idx in range(len(self.persons)):
            if self.persons[idx].face is not None:
                self.persons[idx].face.counter -= 1


    def checkRef(self):
        """Look for the reference faces among the tracked ones."""
        min_similarity = np.inf
        ref_idx = None
        for pi, person in enumerate(self.persons):
            if person.face is not None and person.face.similarity < self.ref_sim_thr and person.face.similarity < min_similarity:
                ref_idx = pi
                min_similarity = person.face.similarity
        # Update the reference person
        if ref_idx is not None:
            self.persons[ref_idx].is_ref = True

    def iterate(self):
        # Fetch the images
        self.lock.acquire()
        try:
            self.image, self.depth = self.cam.getImages()
            self.frame_counter += 1
        except StopIteration:
            self.is_activated = False
            return
        self.lock.release()

        # Step on every person
        self.stepAll()
        self.setPrior()
        # And refresh candidates and persons

    def run(self):
        self.lock.acquire()
        self.image, self.depth = self.cam.getImages()
        self.frame_counter += 1
        self.lock.release()
        self.setPrior()

        if self.debug:
            # The control will be carried by the main thread
            return
        elapsed = np.infty
        while self.is_activated:
            # Control the rate
            if self.frame_counter % 100 == 0:
                print(f'tracker[{self.frame_counter}]:elapsed:{elapsed:.3f} s')
            if elapsed <= PERIOD:
                time.sleep(PERIOD - elapsed)
            start = time.time()
            self.iterate()
            elapsed = time.time() - start
