import numpy as np
import cv2
import rospy
from kobuki_msgs.msg import Sound
from time import sleep
from datetime import datetime

class Motors():

    def __init__(self, motors):
        self.motors = motors
        self.prev_error = 0

        self.scaling_factor = 0.001
        self.Kc = 5
        self.Ki = 2
        self.Kd = 10
        self.K_loss = 3
        self.threshold = 35
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

    def estimateDistance(self, depth, b_box, n_points):
        rand_x = np.random.randint(b_box[0], b_box[2]-1, size=n_points)
        rand_y = np.random.randint(b_box[1], b_box[3]-1, size=n_points)
        random_points = np.column_stack((rand_x, rand_y))
        # Array containing the estimated depths to the person
        depths = []
        for point in random_points:
            [x, y] = point
            depths.append(depth[y, x])
        # Mode calculation
        vals, counts = np.unique(depths, return_counts=True)
        index = np.argmax(counts)
        mode_distance = vals[index]
        # Median calculation
        median_distance = np.median(depths)

        print "Mode: ", mode_distance, ", Median: ", median_distance

        return median_distance



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

        '''
        depth_total = self.network.depth.getImage()
        depth, _, _ = cv2.split(depth_total)
        self.depth = cv2.cvtColor(depth, cv2.COLOR_GRAY2BGR)
        print("Max: %d, min: %d" % (max(self.depth.flatten()), min(self.depth.flatten())))
        '''
        if index is not None: # Found somebody
            box_center = ((box[2] + box[0]) / 2, (box[1] + box[3]) / 2)
            depth_total = self.network.depth.getImage()
            depth, _, _ = cv2.split(depth_total)
            self.depth = cv2.cvtColor(depth, cv2.COLOR_GRAY2BGR)
            print("Max depth: %d, min depth: %d" % (max(self.depth.flatten()), min(self.depth.flatten())))

            # 40 points
            distance = self.estimateDistance(depth, box, 40)


            print('---Person detected, centered on (%d, %d)---' % (box_center[0], box_center[1]))
            h_error = self.center_coords[0] - box_center[0]
            print("Error: %d px" % (h_error))

            P = self.scaling_factor * self.Kc * h_error
            I = self.scaling_factor * self.Ki * (h_error + self.prev_error)
            D = self.scaling_factor * self.Kd * (h_error - self.prev_error)
            w = P + I + D
            print("    P >> %.3f" % (P))
            print("    I >> %.3f" % (I))
            print("    D >> %.3f" % (D))
            print("w = %.3f rad/s" % (w))
            print("")
            print("")
            self.motors.sendW(w)
            # Save values for next iteration
            self.prev_error = h_error
            self.last_center = box_center
            # A person was already detected in the past.
            self.initial = False
            self.can_sound = True
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
            w = self.scaling_factor * self.K_loss * h_loss
            if w < 0:
                print("Person not detected. Turning right (w = %.3f)" % (w))
            else:
                print("Person not detected. Turning left (w = %.3f)" % (w))
            self.motors.sendW(w)
        else: # Nothing detected yet.
            print('---Nothing detected yet.---')
