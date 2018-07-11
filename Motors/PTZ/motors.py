from cprint import *
from Motors import trackers
import numpy as np

class Motors():

    def __init__(self, motors):
        self.motors = motors
        self.limits = self.motors.getLimits()
        self.initial = True

        self.last_center = (0, 0)
        self.threshold = 60

        self.person_tracker = trackers.PersonTracker(same_person_thr=80)

        self.persons = []
        self.faces = []

        self.face_thres = 1.0

    def setCamera(self, cam):
        self.camera = cam



    def setNetworks(self, detection_network, siamese_network):
        self.network = detection_network
        self.siamese_network = siamese_network
        self.person_tracker.setSiameseNetwork(siamese_network)
        self.center_coords = [self.network.original_width/2, self.network.original_height/2]

    def move(self):

        def goToMom(mom_box):
            mom_center = [(mom_box[2] + mom_box[0])/2, (mom_box[1] + mom_box[3])/2]
            current_pos = self.motors.motors.data
            current_pos = [current_pos.pan, current_pos.tilt]

            h_error = mom_center[0] - self.center_coords[0]
            v_error = mom_center[1] - self.center_coords[1]

            if abs(h_error) > self.threshold:
                if mom_center[0] > self.center_coords[0]:
                    cprint.warn('  Horizontal distance: %d. Going right.' % (h_error))
                    h_delta = 1
                else:
                    cprint.warn('  Horizontal distance: %d. Going left.' % (h_error))
                    h_delta = -1
            else:
                cprint.ok('  Horizontal distance: %d (under control).' % (h_error))
                h_delta = 0

            if abs(v_error) > self.threshold:
                if mom_center[1] > self.center_coords[1]:
                    v_delta = -1
                    cprint.warn('  Vertical distance: %d. Going down.' % (v_error))
                else:
                    v_delta = 1
                    cprint.warn('  Vertical distance: %d. Going up.' % (v_error))
            else:
                cprint.ok('  Vertical distance: %d (under control).' % (v_error))
                v_delta = 0


            new_pos = (current_pos[0] + h_delta, current_pos[1] + v_delta)


            self.motors.setPTMotorsData(new_pos[0], new_pos[1], self.limits.maxPanSpeed, self.limits.maxTiltSpeed)
            self.last_center = mom_center



        full_image = self.camera.getImage()
        self.detection_boxes = self.network.boxes
        self.detection_scores = self.network.scores

        self.persons = self.person_tracker.evalPersons(self.detection_boxes, self.detection_scores, full_image)
        print ""
        self.faces = self.person_tracker.getFaces(full_image)

        cprint.info('\t........%d/%d faces detected........' % (len(self.faces), len(self.persons)))

        mom_found_now = False
        # Iteration over all faces and persons...
        for idx in range(len(self.persons)):
            person = self.persons[idx]
            if person.is_mom:
                self.mom_coords = person.coords
                mom_found_now = True
                break
            else:
                faces = person.ftrk.tracked_faces
                if len(faces) > 0:
                    face = faces[0]
                    [f_width, f_height] = [face[2] - face[0], face[3] - face[1]]
                    f_total_box = np.zeros(4, dtype=np.int16)
                    f_total_box[:2] = person[:2] + face[:2]
                    f_total_box[2:4] = f_total_box[:2] + [f_width, f_height]
                    cropped_face = full_image[f_total_box[1]:f_total_box[3], f_total_box[0]:f_total_box[2], :]
                    # We compute the likelihood with mom...
                    dist_to_mom = self.siamese_network.distanceToMom(cropped_face)
                    if dist_to_mom < self.face_thres:
                        # Unset other moms
                        for idx2 in range(len(self.persons)):
                            self.person_tracker.tracked_persons[idx2].is_mom = False
                        # And set that person to mom.
                        self.person_tracker.tracked_persons[idx].is_mom = True
                        self.mom_coords = person.coords
                        mom_found_now = True
                        break
        if mom_found_now:
            cprint.ok("\t\t  Mom found")
            goToMom(self.mom_coords)
        else:
            cprint.warn("\t\t  Looking for mom...")
