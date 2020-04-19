import numpy as np
from utils import distanceBetweenBoxes, bb1inbb2

class Face:
    '''Instance of a tracked face.'''
    def __init__(self, coords, counter, similarity):
        self.coords = coords
        self.counter = counter
        self.similarity = similarity


class FaceTracker:
    '''Keep track of faces inside a person.'''
    def __init__(self, patience=3, same_face_thr=70):
        self.patience = patience
        self.same_face_thr = same_face_thr

        self.faces = []
        self.candidates = []


    def handleDetection(self, detection, similarity):
        '''Update the position of a detected face if it is found to be
        near enough one of the new ones. Otherwise, insert it as a candidate.'''

        found = False
        # Look for matches in the tracked faces
        for idx, face in enumerate(self.faces):
            if distanceBetweenBoxes(face.coords, detection) < self.same_face_thr:
                # Update the face
                self.faces[idx].coords = detection
                self.faces[idx].counter = self.patience
                self.faces[idx].similarity = similarity
                found = True
                break

        if not found:
            # Look into the candidates now
            for idx, cand in enumerate(self.candidates):
                if distanceBetweenBoxes(cand.coords, detection) < self.same_face_thr:
                    # Update the candidate
                    self.candidates[idx].coords = detection
                    self.candidates[idx].counter += 2
                    self.candidates[idx].similarity = similarity
                    found = True
                    break

        if not found:
            # The face is a new candidate
            c = Face(detection, 0, similarity)
            self.candidates.append(c)

        # Refresh everything and return
        self.refresh()
        return self.faces


    def refresh(self):
        '''Refresh the collections (in order to apply the patience).'''
        new_faces = []
        new_candidates = []
        # Candidates:
        for cand in self.candidates:
            if cand.counter >= 0:
                # Survive
                if cand.counter >= self.patience:
                    # The candidate will be a tracked face
                    cand.counter = self.patience
                    new_faces.append(cand)
                else:
                    # Still candidate
                    cand.counter -= 1
                    new_candidates.append(cand)
        self.candidates = new_candidates

        # Faces:
        for face in self.faces:
            if face.counter >= 0:
                # Survive

                face.counter -= 1
                new_faces.append(face)
        self.faces = new_faces



class Person:
    '''Instance of a tracked person.'''
    def __init__(self, coords, counter=0, ftrk=None, is_ref=False):
        self.coords = coords
        self.counter = counter

        if ftrk is not None:
            self.ftrk = ftrk
        else:
            self.ftrk = FaceTracker()

        self.is_ref = is_ref

    def __repr__(self):
        return str(self.coords + [self.counter, self.is_ref])


class PersonTracker:
    '''This class stores the current persons on scene, and refreshes them using
    the latest inferences.'''

    def __init__(self, patience, ref_sim_thr, same_person_thr):
        # Placeholders
        self.persons = []
        self.candidates = []
        self.tracked_counter = 0
        # Parameters
        self.same_person_thr = same_person_thr
        self.ref_sim_thr = ref_sim_thr
        self.patience = patience

    def handleDetections(self, detections):
        '''Update the position of a detected person if it is found to be
        near enough one of the passed ones. Otherwise, insert it as a candidate.'''

        for det in detections:
            found = False
            # Look for matches in the tracked persons
            for idx, person in enumerate(self.persons):
                if distanceBetweenBoxes(person.coords, det) < self.same_person_thr:
                    # Update the person
                    self.persons[idx].coords = det
                    self.persons[idx].counter = self.patience
                    found = True
                    break

            if not found:
                # Look into the candidates now
                for idx, cand in enumerate(self.candidates):
                    if distanceBetweenBoxes(cand.coords, det) < self.same_person_thr:
                        # Update the candidate
                        self.candidates[idx].coords = det
                        self.candidates[idx].counter += 2
                        found = True
                        break

            if not found:
                # The person is a new candidate
                self.tracked_counter += 1
                c = Person(det)
                self.candidates.append(c)

        # Refresh everything and return
        self.refresh()
        return self.persons


    def refresh(self):
        '''Update the stored persons.'''
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


    def handleFaces(self, faces, similarities, delete):
        '''Check if a detected face belongs (spatially) to a person, and track it. Discard it otherwise.'''
        for face, sim in zip(faces, similarities):
            face_std = [face[0]-face[2]//2, face[1]-face[3]//2, face[2], face[1]]
            for person in self.persons:
                if bb1inbb2(face_std, person.coords):
                    # The face belongs to this person. We update it.
                    person.ftrk.handleDetection(face, sim)

    def checkRef(self):
        '''Look for the reference faces among the tracked ones.'''
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

