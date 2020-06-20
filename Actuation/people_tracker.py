import cv2
import threading
import time
from datetime import datetime
from cprint import cprint
from Actuation.tracking_classes import *

FEATURE_PARAMS = dict(maxCorners=100,
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

    def __init__(self, patience, ref_sim_thr, same_person_thr):
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
        self.faces = []
        self.similarities = []

        self.is_activated = True
        self.lock = threading.Lock()

    def setCam(self, cam):
        self.cam = cam

    def setPrior(self):
        """Set the first image on the tracker."""
        self.gray_image = cv2.cvtColor(self.image, cv2.COLOR_RGB2GRAY)
        keypoints = cv2.goodFeaturesToTrack(self.gray_image, **FEATURE_PARAMS)
        self.keypoints = keypoints.squeeze()


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
        # print(f"found {len(found_idx)}/{len(self.keypoints)} keypoints in the image")
        old_found = self.keypoints[found_idx]
        new_found = new_kps[found_idx]

        # And compute the individual displacements for each person
        for candidate in self.candidates:
            # print('iterating on candidate')
            candidate.step(old_found, new_found)
            # print('finished candidate')
        for person in self.persons:
            # print('iterating on person')
            person.step(old_found, new_found)
            # print('finished person')

        # Update the reference frame and keypoints
        self.gray_image = new_image
        self.keypoints = new_kps

    def updateWithDetections(self, boxes, faces, similarities):
        """Reassign the person to the most suitable bounding box."""
        # TODO: is it necessary to call stepAll here?
        for box in boxes:
            pers_distances = np.array(list(map(lambda x: distanceBetweenBoxes(box, x.coords), self.persons)))
            cand_distances = np.array(list(map(lambda x: distanceBetweenBoxes(box, x.coords), self.candidates)))
            # print("distances", pers_distances, cand_distances)
            # Assign to the nearest person, or create one if required
            near_pers = np.where(pers_distances <= self.same_person_thr)[0]
            near_cand = np.where(cand_distances <= self.same_person_thr)[0]
            # print("near_cand", near_cand, len(near_cand))
            # print("near_pers", near_pers, len(near_pers))
            if len(near_pers) > 0 and len(near_cand) == 0:
                # print('to a person')
                # Closest close person
                lowest_idx = np.argmin(pers_distances[near_pers])
                if isinstance(lowest_idx, np.ndarray):
                    lowest_idx = lowest_idx[0]
                self.persons[lowest_idx].coords = box
                # The person is still found
                self.persons[lowest_idx].counter = self.patience
            elif len(near_cand) > 0 and len(near_pers) == 0:
                # print('to a candidate')
                # Closest close candidate
                lowest_idx = np.argmin(cand_distances[near_cand])
                if isinstance(lowest_idx, np.ndarray):
                    lowest_idx = lowest_idx[0]
                self.candidates[lowest_idx].coords = box
                # The candidate is still found
                self.candidates[lowest_idx].counter += 2
            elif len(near_cand) > 0 and len(near_pers) > 0:
                # print('both nearby')
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
                # print('new candidate')
                # This detection can't be assigned to anyone. We create a new candidate
                candidate = Person(box)
                self.candidates.append(candidate)
        self.faces = faces
        self.similarities = similarities
        self.handleFaces()
        self.checkRef()


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

    def handleFaces(self):
        faces = self.faces
        similarities = self.similarities
        """Check if a detected face belongs (spatially) to a person, and track it. Discard it otherwise."""
        for face, sim in zip(faces, similarities):
            face_std = [face[0] - face[2] // 2, face[1] - face[3] // 2, face[2], face[1]]
            for person in self.persons:
                if bb1inbb2(face_std, person.coords):
                    # The face belongs to this person. We update it.
                    person.ftrk.handleDetection(face, sim)

    def checkRef(self):
        """Look for the reference faces among the tracked ones."""
        min_similarity = np.inf
        ref_idx = None
        for pi, person in enumerate(self.persons):
            for fi, face in enumerate(person.ftrk.faces):
                if face.similarity < self.ref_sim_thr and face.similarity < min_similarity:
                    ref_idx = (pi, fi)
                    min_similarity = face.similarity
        # Update the reference person
        if ref_idx is not None:
            for pi in range(len(self.persons)):
                if pi == ref_idx[0]:
                    self.persons[pi].is_ref = True
                else:
                    self.persons[pi].is_ref = False

    def run(self):
        self.lock.acquire()
        self.image, self.depth = self.cam.getImages()
        self.lock.release()
        self.setPrior()
        elapsed = np.infty
        counter = 0
        while self.is_activated:
            # Control the rate
            print(f'tracker[{counter}]:elapsed:{elapsed:.3f} s')
            if elapsed <= PERIOD:
                time.sleep(PERIOD - elapsed)
            start = time.time()
            # Fetch the images
            self.lock.acquire()
            try:
                self.image, self.depth = self.cam.getImages()
            except StopIteration:
                self.is_activated = False
                break
            self.lock.release()

            # Step on every person
            # print('-------')
            # print('before:', [p.counter for p in self.candidates], )
            # print(f"before {len(self.candidates)} candidates, {len(self.persons)} persons")
            self.stepAll()
            # print(f"after {len(self.candidates)} candidates, {len(self.persons)} persons")
            # print("========")
            # And refresh candidates and persons
            self.refresh()

            elapsed = time.time() - start
            # print("elapsed", elapsed)
            counter += 1
