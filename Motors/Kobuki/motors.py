import numpy as np
import cv2
import rospy
from kobuki_msgs.msg import Sound
from time import sleep
from datetime import datetime
from Motors.Kobuki.pid_motors import PIDCommander
from cprint import *


class Motors():
    ''' Class to process the error computed from the RGB and depth images, and send commands,
    which will be intelligently interpreted. '''

    def __init__(self, motors):
        self.motors = motors
        # PID controllers:
        self.w_PID = PIDCommander(
                         func=self.motors.sendW,
                         Kc=7,
                         Ki=0.5,
                         Kd=10,
                         K_loss=1,
                         scaling_factor=0.0003,
                         limiter=1)

        self.v_PID = PIDCommander(
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

        self.last_center = [0, 0]

        # Status flags:
        self.initial = True
        self.can_sound = True
        self.lost = True

        # Interface for turtlebot sounds
        self.sound_publisher = rospy.Publisher('/mobile_base/commands/sound', Sound, queue_size=10)
        # For the publisher to establish:
        sleep(0.5)
        # init_node not necessary (already called by comm)

    def setNetworks(self, detection_network, siamese_network):
        self.network = detection_network
        self.siamese_network = siamese_network
        self.center_coords = (self.network.original_width/2, self.network.original_height/2)

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
        ''' Method called on each iteration. Processes the error and moves the robot. '''
        print("")
        try:
            index = self.network.predictions.index('person')
            box = self.network.boxes[index]
            # predictions are sorted in score order
            # .index returns the lowest position
            # hence, we will keep the most confident bounding box
        except ValueError:
            index = None
            box = [0, 0, 0, 0]

        if index is not None: # Found somebody
            cprint.info('           ---Person detected---')



            ##################################################
            #################### DISTANCE ####################
            ##################################################
            depth_total = self.depth.getImage()
            depth, _, _ = cv2.split(depth_total)
            self.depth_img = cv2.cvtColor(depth, cv2.COLOR_GRAY2BGR)

            # We crop the person, and compute its depth
            person_slice = depth[box[1]:box[3], box[0]:box[2]]
            try:
                distance, grid = self.estimateDistance(person_slice)
            except Exception as e:
                cprint.fatal(e)
                distance = self.v_center

            error = distance - self.v_center

            if error < -self.v_margin:
                # Too far
                v = self.v_PID.processError(error)
                cprint.warn('  Distance: %d px (too near) >> VX = %.3f m/s' % (distance, v))

            elif error > self.v_margin:
                # Too near
                v = self.v_PID.processError(error)
                cprint.warn('  Distance: %d px (too far) >> VX = %.3f m/s' % (distance, v))

            else:
                # Inside range (OK)
                cprint.ok('  Distance: %d px (under control)' % (distance))
                self.v_PID.resetError()
                self.v_PID.brake()




            ###################################################
            ###################### ANGLE ######################
            ###################################################
            box_center = ((box[2] + box[0]) / 2, (box[1] + box[3]) / 2)
            # Compute the horizontal error between the centers (image vs bounding box)
            h_error = self.center_coords[0] - box_center[0]

            if abs(h_error) > self.w_margin:
                # Turning...
                w = self.w_PID.processError(h_error, verbose=True)
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

            # Updating parameters for next iteration
            self.last_center = box_center


            # A person was already detected in the past.
            self.initial = False
            self.can_sound = True
            # If coming back from lost state:
            if self.lost:
                self.sound_publisher.publish(Sound.RECHARGE)
                self.lost = False

        elif not self.initial: # Person lost
            self.lost = True

            # Beeps from the kobuki
            if self.can_sound:
                self.sound_publisher.publish(Sound.BUTTON)
                self.can_sound = False
                self.sounded = datetime.now()
            else:
                elapsed = datetime.now() - self.sounded
                if elapsed.seconds >= 2:
                    self.can_sound = True

            # Slowly turn on the last known direction
            w = self.w_PID.lostResponse()
            # Stops the robot (K_loss = 0)
            v = self.v_PID.lostResponse()
            if w < 0:
                dir = 'right'
            else:
                dir = 'left'

            cprint.fatal("  Person lost. Turning %s (w = %.3f r/s)" % (dir, w))

        else: # Nothing detected yet.
            cprint.warn('           ---Nothing detected yet---')
