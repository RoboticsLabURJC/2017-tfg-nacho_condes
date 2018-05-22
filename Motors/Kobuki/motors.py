import numpy as np
class Motors():

    def __init__(self, motors):
        self.motors = motors
        self.prev_error = 0

        self.scaling_factor = 0.001
        self.Kc = 5
        self.Ki = 2
        self.Kd = 10
        self.threshold = 35

    def setNetwork(self, network):
        self.network = network
        self.center_coords = (self.network.original_width/2, self.network.original_height/2)

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

        box_center = ((box[2] + box[0]) / 2, (box[1] + box[3]) / 2)


        if index is not None:
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
            self.motors.sendW(w)
            self.prev_error = h_error
            print("")
            print("")
