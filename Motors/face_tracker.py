import numpy as np



class Face:
    def __init__(self, coords, last_seen=0):
        self.coords = coords
        self.last_seen = last_seen


class FaceTracker:
    '''
    Class to implement a spatial tracking of the faces detected in the image.
    These faces come already filtered, remaining only those with a highest
    position inside each person.
    '''

    def __init__(self, patience, near_thres=50):
        self.patience = patience
        self.near_thres = near_thres
        self.tracked_faces = []



    def refreshFaces(self):
        ''' We remove old faces from the tracked faces list. '''
        aux_faces = []

        for t_face in self.tracked_faces:
            if t_face.last_seen >= 1:
                new_face = Face(t_face.coords, t_face.last_seen - 1)
                aux_faces.append(new_face)

        self.tracked_faces = aux_faces


    def evalFaces(self, cand_faces):
        '''
        @type cand_faces: list
        @param cand_faces: list containing the candidate face boxes
        '''
        print "n_faces: %d" % (len(self.tracked_faces))
        for c_face in cand_faces:
            # We contrast the new faces with existing tracked faces.
            found = False
            for idx in range(len(self.tracked_faces)):
                t_face = self.tracked_faces[idx]
                if self.pxBetweenBoxes(c_face, t_face.coords) < self.near_thres:
                    # We found a nearby face, which is updated
                    # with the new coords and fresh counter.
                    self.tracked_faces[idx] = Face(c_face, self.patience)
                    found = True
                    break
            if not found:
                # Face not present. We just append it.
                self.tracked_faces.append(Face(c_face, self.patience))

        # Now, we wipe too old faces (not seen in a long time).
        self.refreshFaces()
        return self.tracked_faces


    def pxBetweenBoxes(self, bb1, bb2):
        '''
        This function returns the distance (in px) between two
        bounding boxes (faces), to judge wether they are
        approximately in the same place.
        '''
        print "--------------------------------"
        bb1 = bb1.astype(np.int32)
        bb2 = bb2.astype(np.int32)
        center1 = np.divide([bb1[3] + bb1[1], bb1[2] + bb1[0]], 2)
        center2 = np.divide([bb2[3] + bb2[1], bb2[2] + bb2[0]], 2)
        aux = np.sum(np.square(np.subtract(center1, center2)))

        distance = np.sqrt(np.sum(np.square(np.subtract(center1, center2))))
        return distance
