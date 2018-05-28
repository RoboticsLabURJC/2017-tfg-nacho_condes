import numpy as np
import cv2
import rospy
from kobuki_msgs.msg import Sound
from time import sleep
from datetime import datetime
import matplotlib.pyplot as plt

from cprint import *

class PID:
    def __init__(self, Kc, Ki, Kd, K_loss, scaling_factor, limiter, prev_pct):
        self.Kc = Kc
        self.Ki = Ki
        self.Kd = Kd
        self.K_loss = K_loss
        self.scaling_factor = scaling_factor
        self.prev_error = 0
        self.cum_error = 0
        self.limiter = limiter
        # To avoid abrupt stops, we combine with the previous input
        self.prev_pct = prev_pct

    def resetCounter(self):
        ''' Reset the cumulative error when the target is in range. '''
        self.cum_error = 0

    def compute_response(self, error, detail=False):
        # Error "echo" to soften output
        error = error + self.prev_pct * self.prev_error
        P = self.Kc * error
        I = self.Ki * (error + self.cum_error)
        D = self.Kd * (error - self.prev_error)
        #response = self.scaling_factor * (P + I + D)
        response = P + I + D
        if detail:
            cprint.ok("    P >> %.3f" % (P))
            cprint.ok("    I >> %.3f" % (I))
            cprint.ok("    D >> %.3f" % (D))
        ''' Current error storage... '''
        self.prev_error = error
        self.cum_error += error

        scaled_response = response * self.scaling_factor
        if scaled_response > self.limiter:
            scaled_response = self.limiter

        return scaled_response

    def loss_response(self, last_error=0):
        response = self.scaling_factor * self.K_loss * last_error
        return response

class Motors():

    def __init__(self, motors):
        self.motors = motors

        self.w_PID = PID(Kc=2,
                         Ki=1,
                         Kd=10,
                         K_loss=10,
                         scaling_factor=0.0003,
                         limiter=1,
                         prev_pct = 0.7)

        self.v_PID = PID(Kc=3,
                         Ki=2,
                         Kd=10,
                         K_loss=0,
                         scaling_factor=0.0001,
                         limiter=0.5,
                         prev_pct = 0.7)

        self.w_threshold = 70
        self.v_thresholds = [30, 50]
        self.last_center = [0, 0]
        self.initial = True
        self.can_sound = True
        self.lost = True

        # Interface for turtlebot sounds
        self.sound_publisher = rospy.Publisher('/mobile_base/commands/sound', Sound, queue_size=10)
        # For the publisher to establish:
        sleep(0.5)
        # init_node not necessary (already called by comm)

    def setNetwork(self, network):
        self.network = network
        self.center_coords = (self.network.original_width/2, self.network.original_height/2)

    def estimateDistance(self, person):
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
        except Exception as e:
            cprint.fatal(e)
            samples = 0
        median = np.median(samples)

        # Returning also the grid, to be able to draw it on the depth image
        return median, grid

    def move(self):
        try:
            index = self.network.predictions.index('person')
            # predictions are sorted in score order
            # .index returns the lowest position
            # hence, we will keep the most confident bounding box
            box = self.network.boxes[index]
        except AttributeError:
            index = None
            box = [0, 0, 0, 0]
        except ValueError:
            index = None
            box = [0, 0, 0, 0]
        if index is not None: # Found somebody
            cprint.info('---Person detected---')
            ##################################################
            #################### DISTANCE ####################
            ##################################################
            depth_total = self.network.depth.getImage()
            depth, _, _ = cv2.split(depth_total)
            self.depth = cv2.cvtColor(depth, cv2.COLOR_GRAY2BGR)

            # We crop the person, and compute its depth
            person_slice = depth[box[1]:box[3], box[0]:box[2]]

            distance, grid = self.estimateDistance(person_slice)
            if distance > self.v_thresholds[1]:
                vx = self.v_PID.compute_response(distance)
                cprint.warn('  Distance: %d px (too far) == VX: %.3f' % (distance, vx))
                self.motors.sendVX(vx)
            elif distance < self.v_thresholds[0] and distance > 0:
                vx = self.v_PID.compute_response(1000/distance, True)
                cprint.warn('  Distance: %d px (too near) == VX: %.3f' % (distance, -vx))
                self.motors.sendVX(-vx)
            else:
                cprint.ok('  Distance: %d px (under control)' % (distance))
                self.v_PID.resetCounter()
                v = self.v_PID.compute_response(0)
                self.motors.sendVX(v)


            ###################################################
            ###################### ANGLE ######################
            ###################################################
            box_center = ((box[2] + box[0]) / 2, (box[1] + box[3]) / 2)
            h_error = self.center_coords[0] - box_center[0]
            if abs(h_error) > self.w_threshold:
                w = self.w_PID.compute_response(h_error, True)

                cprint.warn('  Angle: %d px == W: %.3f' % (h_error, w))
                self.motors.sendW(w)
            else:
                cprint.ok('  Angle: %d px (under control)' % (h_error))
                self.w_PID.resetCounter()
                v = self.v_PID.compute_response(0)
                self.motors.sendW(v)

            self.last_center = box_center
            # A person was already detected in the past.
            self.initial = False
            self.can_sound = True
            # If coming back from lost state:
            if self.lost:
                self.sound_publisher.publish(Sound.RECHARGE)
                self.lost = False


        elif not self.initial: # Person lost
            last_x = self.last_center[0]
            self.lost = True

            if self.can_sound:
                self.sound_publisher.publish(Sound.BUTTON)
                self.can_sound = False
                self.sounded = datetime.now()
            else:
                elapsed = datetime.now() - self.sounded
                if elapsed.seconds >= 2:
                    self.can_sound = True

            h_loss = self.center_coords[0] - last_x
            w = self.w_PID.loss_response(h_loss)
            v = self.v_PID.loss_response()

            if w < 0:
                cprint.fatal("  Person lost. Turning right (w = %.3f)" % (w))
            else:
                cprint.fatal("  Person lost. Turning left (w = %.3f)" % (w))
            self.motors.sendW(w)
            self.motors.sendVX(v)
        else: # Nothing detected yet.
            cprint.warn('---Nothing detected yet...---')
