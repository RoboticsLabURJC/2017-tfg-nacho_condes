import numpy as np
import cv2
import rospy
from kobuki_msgs.msg import Sound
from time import sleep
from datetime import datetime
from Motors.Kobuki.pid_driver import PIDDriver
from cprint import cprint

from Motors import talk

class Motors():
    ''' Class to process the error computed from the RGB and depth images, and send commands,
    which will be intelligently interpreted. '''

    def __init__(self, motors):
        self.motors = motors

        self.faces = None
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
                         Ki=0.1,
                         Kd=3,
                         K_loss=0,
                         scaling_factor=0.005,
                         limiter=0.5)

        # Parameters for error processing:
        self.w_center = 0
        self.w_margin = 60

        self.v_center = 60
        self.v_margin = 10

        # To restore the margin when it is dynamically modified
        self.original_v_margin = self.v_margin


        # Interface for turtlebot sounds
        self.sound_publisher = rospy.Publisher('/mobile_base/commands/sound', Sound, queue_size=10)
        # For the publisher to establish:
        sleep(0.5)
        # init_node not necessary (already called by comm)

        # Person/face processing parameters
        self.mom_box = None
        self.last_mom_box = None
        self.last_known_mom_box = None
        self.face_thres = 1.0
        self.last_det_time = None
        self.patience = 0


        # Overswitching avoidance flag
        self.margin_expanded = False

        self.faces = []
        # Contains the faces boxes, w.r.t. the
        # coordinates of the whole image.
        self.total_faces = []


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
                # Avoid overswitching
                if not self.margin_expanded:
                    self.v_margin = self.v_margin - 5
                    self.margin_expanded = True

                self.margin_expanded = True
                #cprint.info("New margin: %d" % (self.v_margin))

            elif error > self.v_margin:
                # Too near
                v = self.v_PID.processError(error, verbose=False)
                cprint.warn('  Distance: %d px (too far) >> VX = %.3f m/s' % (distance, v))
                # Avoid overswitching
                if not self.margin_expanded:
                    self.v_margin = self.v_margin - 5
                    self.margin_expanded = True
                #cprint.info("New margin: %d" % (self.v_margin))

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
        self.detection_boxes = self.network.boxes
        self.detection_scores = self.network.scores

        num_detections = len(self.detection_boxes)
        # Initial distances (with an imputed high value to avoid false positives).
        coord_distances = np.ones(num_detections) * 1000
        face_distances = np.ones(num_detections) * 1000

        aux_faces = []
        aux_total_faces = []

        # Iteration over each detected person
        for idx in range(num_detections):
            box = self.detection_boxes[idx]
            if self.last_mom_box is not None:
                # Distance to the last known mom position
                coord_distances[idx] = self.distanceBetweenBoxes(box, self.last_mom_box)

            color_person = full_image[box[1]:box[3], box[0]:box[2]]
            face, f_box = self.siamese_network.getFace(color_person)
            # Has it face?
            if face is not None:
                aux_faces.append(face)
                face_distances[idx] = self.siamese_network.distanceToMom(face)

                # We build the total face bounding box
                [f_width, f_height] = [f_box[2] - f_box[0], f_box[3] - f_box[1]]
                f_total_box = np.zeros(4, dtype=np.int16)
                f_total_box[:2] = box[:2] + f_box[:2]
                f_total_box[2:4] = f_total_box[:2] + [f_width, f_height]
                aux_total_faces.append(f_total_box)
                #print("Distance: %.3f" % (face_distances[idx]))

        cprint.info('\t........%d/%d faces detected........' % (len(self.faces), num_detections))
        # Final assignation (to avoid empty values on GUI).
        self.faces = aux_faces
        self.total_faces = aux_total_faces

        potential_idx = face_distances < self.face_thres
        # If potential faces were found,
        # mom will be the closest one to the reference.
        if potential_idx.any():
            min_distance = min(face_distances[potential_idx])
            mom_index = face_distances.tolist().index(min_distance)
            # Save the coordinates.
            self.mom_box = self.detection_boxes[mom_index]
            cprint.ok('\t\t  Mom found!')
            self.patience = 0

            # Sound if mom was lost...
            if self.last_mom_box is None and self.last_det_time is not None:
                now = datetime.now()
                time_lost = now - self.last_det_time
                if time_lost.seconds >= 5:
                    #talk.say_mama()
                    pass

            #goToMom(self.mom_box)
            self.last_mom_box = self.detection_boxes[mom_index]
            self.last_known_mom_box = self.detection_boxes[mom_index]
            self.last_det_time = datetime.now()

        else:
            mom_index = None
            self.mom_box = None
            cprint.warn('\t\tDistances: {}'.format(np.sort(face_distances)))
            '''
            if self.last_mom_box is not None:
                if self.patience < 10:
                    # Mom has just been seen
                    cprint.warn("\t\t\t Heading to the last mom's known position")
                    goToMom(self.last_mom_box)
                    self.last_mom_box = None
                    self.patience += 1
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
            '''
