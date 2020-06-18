import numpy as np
from utils import distanceBetweenBoxes, bb1inbb2, center2Corner




class Face:
    """Instance of a tracked face."""
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
        """Refresh the collections (in order to apply the patience)."""
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
    """Instance of a tracked person."""
    def __init__(self, coords, counter=0, ftrk=None, is_ref=False, im_size=(640, 480)):
        self.coords = coords
        self.counter = counter

        if ftrk is not None:
            self.ftrk = ftrk
        else:
            self.ftrk = FaceTracker()

        self.is_ref = is_ref

        self.prev_crop = None
        self.keypoints = None
        self.im_size = im_size

    def step(self, old_kps, new_kps):
        """Perform a forward step, computing the displacement
        using the descriptors found inside the location of the person."""

        # Look for suitable descriptors, "bounding box" (descriptor point) inside the coords?
        valid_idxs = list(map(lambda kp: bb1inbb2([kp[0], kp[1], 0, 0], self.coords), old_kps))
        if sum(valid_idxs) == 0:
            return

        old_valid = old_kps[valid_idxs]
        new_valid = new_kps[valid_idxs]

        # And compute the average displacement
        displacement = new_valid - old_valid
        avg_displ = displacement.mean(axis=0)

        # TODO: soften displacement usign the last N iterations

        # Move the bounding box accordingly (keeping it inside the image)
        self.coords[0] = np.clip(self.coords[0] + avg_displ[0], 0, self.im_size[0])
        self.coords[1] = np.clip(self.coords[1] + avg_displ[1], 0, self.im_size[1])

        self.counter += 2


    def __repr__(self):
        return str(self.coords + [self.counter, self.is_ref])
