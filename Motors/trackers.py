import numpy as np


def pxBetweenBoxes(bb1, bb2):
    '''
    This function returns the distance (in px) between two
    bounding boxes, to judge wether they are
    approximately in the same place.
    '''
    bb1 = bb1.astype(np.int32)
    bb2 = bb2.astype(np.int32)
    center1 = np.divide([bb1[3] + bb1[1], bb1[2] + bb1[0]], 2)
    center2 = np.divide([bb2[3] + bb2[1], bb2[2] + bb2[0]], 2)
    aux = np.sum(np.square(np.subtract(center1, center2)))

    distance = np.sqrt(np.sum(np.square(np.subtract(center1, center2))))
    return distance


class Face:
    def __init__(self, coords, counter=0):
        self.coords = coords
        self.counter = counter

    def __getitem__(self, a):
        return self.coords[a]

    def __repr__(self):
        return "Face: " + str(self.coords) + ", counter: %d" % (self.counter)

    def __len__(self):
        return 1


class FaceTracker:
    '''
    Class to implement a spatial tracking of the faces detected in the image.
    These faces come already filtered, remaining only those with a highest
    position inside each person.
    '''

    def __init__(self, patience=3, same_face_thr=70):
        self.patience = patience
        self.same_face_thr = same_face_thr

        self.tracked_faces = []
        self.candidate_faces = []


    def refreshFaces(self):
        aux_tracked = []
        aux_candidate = []
        # Candidate face update...
        for c_face in self.candidate_faces:
            if c_face.counter >= 0:
                # This face survives.
                if c_face.counter == self.patience:
                    # We will track this face from now.
                    aux_tracked.append(Face(c_face.coords, self.patience))
                else:
                    # It will stay as a candidate for now.
                    aux_candidate.append(Face(c_face.coords, c_face.counter - 1))
        # We update the list of candidate faces.
        self.candidate_faces = aux_candidate

        # Tracked faces now...
        # We iterate over each one of them. If they survive,
        # we check its distance to mom's face, and store it.
        for t_face in self.tracked_faces:
            if t_face.counter >= 0:
                # The face survives.
                aux_tracked.append(Face(t_face.coords, t_face.counter - 1))
        self.tracked_faces = aux_tracked



    def evalFaces(self, d_face):
        # We contrast the new faces with existing tracked faces.
        found = False
        if len(d_face) > 0:
            for idx in range(len(self.tracked_faces)):
                t_face = self.tracked_faces[idx]
                if pxBetweenBoxes(d_face, t_face.coords) < self.same_face_thr:
                    # We found a nearby face, which is updated
                    # with the new coords and fresh counter.
                    self.tracked_faces[idx] = Face(d_face, self.patience)
                    found = True
                    break
            # Face not tracked. Is it a candidate?
            for idx in range(len(self.candidate_faces)):
                c_face = self.candidate_faces[idx]
                if pxBetweenBoxes(d_face, c_face.coords) < self.same_face_thr:
                    # Candidate face found in the same place (approximately).
                    # Updating face.
                    self.candidate_faces[idx] = Face(d_face, min(c_face.counter, self.patience) + 2)
                    found = True
                    break
            if not found:
                # Face not present. We just append it.
                self.candidate_faces.append(Face(d_face, 1))

        # Now, we wipe too old faces (not seen in a long time).
        self.refreshFaces()
        return self.tracked_faces


class Person:
    def __init__(self, coords, score, counter=0, f_tracker=None, is_mom=False):
        self.coords = coords
        self.counter = counter
        self.score = score

        if f_tracker is not None:
            self.ftrk = f_tracker
        else:
            self.ftrk = FaceTracker()

        self.is_mom = is_mom

    def __getitem__(self, a):
        return self.coords[a]

    def __repr__(self):
        if self.is_mom:
            what = 'Mom'
        else:
            what = 'Person'
        return what + ": " + str(self.coords) + "(%.2f)" % (self.score) + '. Counter: %d' %(self.counter)


class PersonTracker:
    def __init__(self, patience=5, mom_dist_thr=1.00, same_person_thr=130):
        self.patience = patience
        self.mom_dist_thr = mom_dist_thr
        self.same_person_thr = same_person_thr

        self.candidate_persons = []
        self.tracked_persons = []


    def setSiameseNetwork(self, s_network):
        self.siamese_network = s_network
        FaceTracker.siamese_network = s_network

    def refreshPersons(self):
        aux_tracked = []
        aux_candidate = []
        # Candidate persons update...
        for c_person in self.candidate_persons:
            if c_person.counter >= 0:
                # This person survives.
                if c_person.counter == self.patience:
                    # We will track this person from now.
                    aux_tracked.append(Person(c_person.coords,
                                              c_person.score,
                                              self.patience,
                                              c_person.ftrk))
                else:
                    # It will stay as a candidate for now.
                    aux_candidate.append(Person(c_person.coords,
                                                c_person.score,
                                                c_person.counter - 1,
                                                c_person.ftrk))
        # We update the list of candidate persons.
        self.candidate_persons = aux_candidate

        # Tracked persons now...
        for t_person in self.tracked_persons:
            if t_person.counter >= 0:
                # The person survives.
                aux_tracked.append(Person(t_person.coords,
                                          t_person.score,
                                          t_person.counter - 1,
                                          t_person.ftrk,
                                          t_person.is_mom))
        self.tracked_persons = aux_tracked

    def evalPersons(self, detected_persons, detection_scores, full_image):
        for idx in range(len(detected_persons)):
            d_person = detected_persons[idx]
            d_score = detection_scores[idx]

            found = False
            # We contrast the new person with existing tracked persons.
            for idx in range(len(self.tracked_persons)):
                t_person = self.tracked_persons[idx]
                if pxBetweenBoxes(d_person, t_person.coords) < self.same_person_thr:
                    # Tracked person found in the same place (approximately).
                    # Updating person.
                    self.tracked_persons[idx] = Person(d_person,
                                                       d_score,
                                                       min(t_person.counter, self.patience) + 2,
                                                       t_person.ftrk,
                                                       t_person.is_mom)
                    found = True
                    break

            # Person not tracked. Is it a candidate?
            for idx in range(len(self.candidate_persons)):
                c_person = self.candidate_persons[idx]
                if pxBetweenBoxes(d_person, c_person.coords) < self.same_person_thr:
                    # Candidate person found in the same place (approximately).
                    # Updating person.
                    self.candidate_persons[idx] = Person(d_person,
                                                         d_score, min(c_person.counter, self.patience) + 2,
                                                         c_person.ftrk)
                    found = True
                    break

            if not found:
                # Not present at all. New candidate!
                self.candidate_persons.append(Person(d_person, d_score, 1))

        # Now, we refresh the persons counter, and move them to their new
        # list, if necessary.
        self.refreshPersons()
        return self.tracked_persons

    def getFaces(self, full_image):
        # Broadcast of the image for all the trackers to have it.
        total_faces = []
        for idx in range(len(self.tracked_persons)):
            # We will look for faces in each person.
            person = self.tracked_persons[idx]
            box = person.coords
            width, height = [box[2] - box[0], box[3] - box[1]]
            cropped_person = full_image[box[1]:box[3], box[0]:box[2]]
            face, f_box = self.siamese_network.getFace(cropped_person)
            # Faces were detected, and the highest one was returned
            t_faces = person.ftrk.evalFaces(f_box)
            if len(t_faces) != 0:
                face = t_faces[0]
                # We rewrite the coordinates with respect to
                # the entire image
                [f_width, f_height] = [face[2] - face[0], face[3] - face[1]]
                f_total_box = np.zeros(4, dtype=np.int16)
                f_total_box[:2] = box[:2] + face[:2]
                f_total_box[2:4] = f_total_box[:2] + [f_width, f_height]
                total_faces.append(f_total_box)

        return total_faces
