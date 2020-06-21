import numpy as np
from utils import distanceBetweenBoxes, bb1inbb2, center2Corner


class Face:
    """Instance of a tracked face."""
    def __init__(self, coords, similarity, counter=5):
        self.coords = coords
        self.similarity = similarity
        self.counter = counter


class Person:
    """Instance of a tracked person."""
    def __init__(self, coords, counter=0, face=None, is_ref=False, im_size=(640, 480)):
        self.coords = coords
        self.counter = counter

        self.face = face
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

        # Move the bounding box accordingly (keeping it inside the image)
        self.coords[0] = np.clip(self.coords[0] + avg_displ[0], 0, self.im_size[0])
        self.coords[1] = np.clip(self.coords[1] + avg_displ[1], 0, self.im_size[1])

        # And rescale it accordingly to the distribution of the keypoints
        old_std = old_valid.std(axis=0)
        new_std = new_valid.std(axis=0)
        if np.count_nonzero(old_std) != 0 and np.count_nonzero(new_std):
            std_ratio = new_std / old_std
            self.coords[2] = self.coords[2] * std_ratio[0]
            self.coords[3] = self.coords[3] * std_ratio[1]


        # self.counter += 2

        # Remove the face if it does not belong to the person anymore
        if self.face is not None:
            # self.face.counter -= 1
            if not bb1inbb2(self.face.coords, self.coords):
                self.face = None
            if self.face.counter <= 0:
                self.face = None

    def setFace(self, coords, similarity):
        face = Face(coords, similarity)
        self.face = face


    def __repr__(self):
        thestr = f'{self.coords[:4]} [{self.counter}, {self.is_ref}]'
        if self.face is not None:
            thestr += f' -- Face: {self.face.coords[:4]} ({self.face.counter}, {self.face.similarity})'
        return thestr + '\n'
