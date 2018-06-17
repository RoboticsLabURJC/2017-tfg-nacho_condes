import numpy as np
import cv2
import rospy
from kobuki_msgs.msg import Sound
from time import sleep
from datetime import datetime
from Motors.Kobuki.pid_driver import PIDDriver
from cprint import *

from Motors import talk

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
                         scaling_factor=0.0003,
                         limiter=1)

        self.v_PID = PIDDriver(
                         func=self.motors.sendVX,
                         Kc=2,
                         Ki=0.1,
                         Kd=4,
                         K_loss=0,
                         scaling_factor=0.003,
                         limiter=0.5)

        # Parameters for error processing:
        self.w_center = 0
        self.w_margin = 60

        self.v_center = 40
        self.v_margin = 15


        # Interface for turtlebot sounds
        self.sound_publisher = rospy.Publisher('/mobile_base/commands/sound', Sound, queue_size=10)
        # For the publisher to establish:
        sleep(0.5)
        # init_node not necessary (already called by comm)

        # Person/face processing parameters
        self.mom_box = None
        self.last_mom_box = None
        self.last_known_mom_box = None
        self.face_thres = 1.3
        self.last_det_time = None


    def distanceBetweenBoxes(self, box1, box2):
        '''
        This function returns the distance (in px) between two
        bounding boxes (rectangles), to judge wether they are
        approximately in the same place.
        '''
        center1 = np.divide([box1[3] + box1[1], box1[2] + box1[0]], 2)
        center2 = np.divide([box2[3] + box2[1], box2[2] + box2[0]], 2)
        distance = np.sqrt(np.sum(np.square(np.subtract(center1, center2))))

        return distance

    def setNetworks(self, detection_network, siamese_network):
        self.network = detection_network
        self.siamese_network = siamese_network
        self.center_coords = (self.network.original_width/2, self.network.original_height/2)


    def setCamera(self, cam):
        self.camera = cam

    def setDepth(self, depth):
        self.depth = depth

    def estimateDistance(self, person):
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
        samples = person[grid]
        median = np.median(samples)

        # Returning also the grid, to be able to draw it on the depth image (TODO)
        return median, grid

    def move(self):
        '''
        Method called on each iteration. Processes the error and moves the robot.
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
            distance, grid = self.estimateDistance(mom_depth)
            # V error processing (go forward/backward)
            error = distance - self.v_center
            if error < -self.v_margin:
                # Too far
                v = self.v_PID.processError(error, verbose=False)
                cprint.warn('  Distance: %d px (too near) >> VX = %.3f m/s' % (distance, v))

            elif error > self.v_margin:
                # Too near
                v = self.v_PID.processError(error, verbose=False)
                cprint.warn('  Distance: %d px (too far) >> VX = %.3f m/s' % (distance, v))

            else:
                # Inside range (OK)
                cprint.ok('  Distance: %d px (under control)' % (distance))
                self.v_PID.resetError()
                self.v_PID.brake()

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
                    dir = 'right'
                else:
                    dir = 'left'

                cprint.warn('  Angle: %d px >> Turning %s (w: %.3f r/s)' % (h_error, dir, w))
            else:
                # Inside the angle margin (OK)
                cprint.ok('  Angle: %d px (under control)' % (h_error))
                self.w_PID.resetError()
                self.w_PID.brake()



        # Network outputs. Exclusively high score people detections.
        detection_boxes = self.network.boxes
        num_detections = len(detection_boxes)
        # Initial distances (with an imputed high value to avoid false positives).
        coord_distances = np.ones(num_detections) * 1000
        face_distances = np.ones(num_detections) * 1000
        n_faces = 0

        # Iteration over each detected person
        for idx in range(num_detections):
            box = detection_boxes[idx]
            if self.last_mom_box is not None:
                # Distance to the last known mom position
                coord_distances[idx] = self.distanceBetweenBoxes(box, self.last_mom_box)

            color_person = full_image[box[1]:box[3], box[0]:box[2]]
            face = self.siamese_network.getFace(color_person)
            # Has it face?
            if face is not None:
                n_faces += 1
                face_distances[idx] = self.siamese_network.distanceToMom(face)

        cprint.info('\t........%d/%d faces detected........' % (n_faces, num_detections))


        potential_idx = face_distances < self.face_thres
        # If potential faces were found,
        # mom will be the closest one to the reference.
        if potential_idx.any():
            min_distance = min(face_distances[potential_idx])
            mom_index = face_distances.tolist().index(min_distance)
            # Save the coordinates.
            self.mom_box = detection_boxes[mom_index]
            cprint.ok('\t\t  Mom found!')

            # Sound if mom was lost...
            if self.last_mom_box is None and self.last_det_time is not None:
                now = datetime.now()
                time_lost = now - self.last_det_time
                if time_lost.seconds >= 5:
                    talk.say_mama()

            goToMom(self.mom_box)
            self.last_mom_box = detection_boxes[mom_index]
            self.last_known_mom_box = detection_boxes[mom_index]
            self.last_det_time = datetime.now()

        else:
            mom_index = None
            self.mom_box = None
            if self.last_mom_box is not None:
                # Mom has just been seen
                cprint.warn("\t\t\t Heading to the last mom's known position")
                goToMom(self.last_mom_box)
                self.last_mom_box = None
            else:
                if self.last_known_mom_box is not None:
                    # Mom is lost.
                    w = self.w_PID.lostResponse()
                    v = self.v_PID.lostResponse()
                    if w < 0:
                        dir = 'right'
                    else:
                        dir = 'left'
                    cprint.err('\t\t\tMom lost. Turning %s...' % (dir))
                else:
                    # I've never seen mom, Hulio.
                    cprint.warn('\t\t\tWaiting for mom...')
