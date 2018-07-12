import numpy as np
import cv2
import rospy
from kobuki_msgs.msg import Sound
from time import sleep
from datetime import datetime
from Motors.Kobuki.pid_driver import PIDDriver
from cprint import cprint

from Motors import talk

from Motors import trackers

class Motors():
    ''' Class to process the error computed from the RGB and depth images, and send commands,
    which will be intelligently interpreted. '''

    def __init__(self, motors):
        self.motors = motors

        # PID controllers:
        self.w_PID = PIDDriver(
                         func=self.motors.sendW,
                         Kc=7,
                         Ki=0.5,
                         Kd=10,
                         K_loss=1,
                         scaling_factor=0.0005,
                         limiter=1)

        self.v_PID = PIDDriver(
                         func=self.motors.sendVX,
                         Kc=2,
                         Ki=0.08,
                         Kd=2.7,
                         K_loss=0,
                         scaling_factor=0.005,
                         limiter=0.7)

        # Parameters for error processing:
        self.w_center = 0
        self.w_margin = 50

        self.v_center = 40
        self.v_margin = 10
        # Overswitching avoidance flag
        self.margin_expanded = False
        # To restore the margin when it is dynamically modified
        self.original_v_margin = self.v_margin

        self.face_thres = 0.9

        self.mom_coords = None
        self.prev_error = None
        self.persons = []
        self.faces = []

        # Interface for turtlebot sounds
        #self.sound_publisher = rospy.Publisher('/mobile_base/commands/sound', Sound, queue_size=10)
        # For the publisher to establish:
        #sleep(0.5)
        # init_node not necessary (already called by comm)

        self.person_tracker = trackers.PersonTracker()

    def setNetworks(self, detection_network, siamese_network):
        self.network = detection_network
        self.siamese_network = siamese_network
        self.person_tracker.setSiameseNetwork(siamese_network)
        self.center_coords = (self.network.original_width/2, self.network.original_height/2)

    def setCamera(self, cam):
        self.camera = cam

    def setDepth(self, depth):
        self.depth = depth

    def estimateDepth(self, person):
        ''' Given a depth bounding box, estimate the distance to the person which is inside of it. '''
        height, width = person.shape
        # Firstly, we crop the 10% outer part of the box
        vert_crop, horz_crop = [height / 10, width / 10]
        cropped = person[vert_crop:-vert_crop, horz_crop:-horz_crop]
        c_height, c_width = cropped.shape
        # Now we compute the 10x10 grid to sample depth inside
        # the person. We define it as 16 bit long because it can be
        # bigger than 255
        resolution = 10
        v_grid, h_grid = [np.linspace(0, c_height-1, num=resolution, dtype=np.uint16),
                          np.linspace(0, c_width-1, num=resolution, dtype=np.uint16)]

        grid = np.meshgrid(v_grid, h_grid)
        # We sample the depth map
        try:
            samples = person[grid]
        except:
            # Sometimes the depth image cannot be retrieved (TODO)
            samples = 0
        median = np.median(samples)

        # Returning also the grid, to be able to draw it on the depth image (TODO)
        return median, grid

    def move(self):
        '''
        Method called on each iteration. Detects persons and look for mom.
        Commands the robot towards mom if it is found.
        '''
        # We get the full RGB and D images.
        full_image = self.camera.getImage()
        d = self.depth.getImage()
        full_depth, _, _ = cv2.split(d)

        def goToMom(mom_box):
            '''
            Function to go towards mom.
            '''
            # We compute the distance to mom
            ################################
            ############### v ##############
            ################################
            mom_depth = full_depth[mom_box[1]:mom_box[3], mom_box[0]:mom_box[2]]
            distance, grid = self.estimateDepth(mom_depth)
            # V error processing (go forward/backward)
            error = distance - self.v_center
            if self.prev_error is not None:
                d_error = abs(error - self.prev_error)
            else:
                d_error = 0
            # Avoid jumps
            if d_error < 10.0:
                if error < -self.v_margin:
                    # Too near
                    v = self.v_PID.processError(error, verbose=True)
                    cprint.warn('  Distance: %d px (too near) >> VX = %.3f m/s' % (distance, v))
                    # Avoid overswitching
                    if not self.margin_expanded:
                        self.v_margin = self.v_margin - 5
                        self.margin_expanded = True

                    self.margin_expanded = True
                    cprint.info("New margin: %d" % (self.v_margin))

                elif error > self.v_margin:
                    # Too far
                    v = self.v_PID.processError(error, verbose=True)
                    cprint.warn('  Distance: %d px (too far) >> VX = %.3f m/s' % (distance, v))
                    # Avoid overswitching
                    if not self.margin_expanded:
                        self.v_margin = self.v_margin - 5
                        self.margin_expanded = True
                    cprint.info("New margin: %d" % (self.v_margin))

                else:
                    # Inside range (OK)
                    cprint.ok('  Distance: %d px (under control)' % (distance))
                    self.v_PID.resetError()
                    self.v_PID.brake()

                    if self.margin_expanded and error < self.original_v_margin:
                        # The margin can be restored...
                        cprint.info("Margin restored.")
                        self.v_margin = self.original_v_margin
                        self.margin_expanded = False

            self.prev_error = error

            # Now, we compute the necessary turning
            ################################
            ############### w ##############
            ################################

            mom_center = (mom_box[2] + mom_box[0]) / 2
            h_error = self.center_coords[0] - mom_center

            if abs(h_error) > self.w_margin:
                # Turning...
                w = self.w_PID.processError(h_error, verbose=False)
                if w < 0:
                    turn_dir = 'right'
                else:
                    turn_dir = 'left'

                cprint.warn('  Angle: %d px >> Turning %s (w: %.3f r/s)' % (h_error, turn_dir, w))
            else:
                # Inside the angle margin (OK)
                cprint.ok('  Angle: %d px (under control)' % (h_error))
                self.w_PID.resetError()
                self.w_PID.brake()

        # Network outputs. Exclusively high score people detections.
        self.detection_boxes = self.network.boxes
        self.detection_scores = self.network.scores

        # num_detections = len(self.detection_boxes)
        # We retrieve every detected face on the current frame.
        self.persons = self.person_tracker.evalPersons(self.detection_boxes, self.detection_scores, full_image)
        # Now, we look for faces in those persons.
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

        # If mom is being tracked, we move the robot towards it.
        if mom_found_now:
            cprint.ok("\t\t  Mom found")
            goToMom(self.mom_coords)
        else:
            cprint.warn("\t\t  Looking for mom...")
            self.v_PID.lostResponse()
            self.w_PID.lostResponse()
