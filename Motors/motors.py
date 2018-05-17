class Motors():

    def __init__(self, motors):
        self.motors = motors
        self.limits = self.motors.getLimits()

    def setNetwork(self, network):
        self.network = network
        self.center_coords = (self.network.original_width/2, self.network.original_height/2)
        self.epsilon = 60

    def moveCam(self):
        try:
            index = self.network.predictions.index('person')
            box = self.network.boxes[index]
        except AttributeError:
            index = None
            box = [0, 0, 0, 0]
        except ValueError:
            index = None
            box = [0, 0, 0, 0]

        box_center = ((box[2] + box[0]) / 2, (box[1] + box[3]) / 2)
        current_pos = self.motors.motors.data
        current_pos = (current_pos.pan, current_pos.tilt)
        if index is not None:
            print('---Person detected, centered on (%d, %d)---' % (box_center[0], box_center[1]))
            print('+++Current position: (%d, %d)+++' % (current_pos[0], current_pos[1]))

            # Horizontal delta:
            if abs(box_center[0] - self.center_coords[0]) > self.epsilon:
                if box_center[0] > self.center_coords[0]:
                    print(" Go right.")
                    h_delta = 2
                else:
                    print(" Go left.")
                    h_delta = -2
            else:
                h_delta = 0

            # Vertical delta
            if abs(box_center[1] - self.center_coords[1]) > self.epsilon:

                if box_center[1] > self.center_coords[1]:
                    v_delta = -2
                    print(" Go down.")
                else:
                    v_delta = 2
                    print(" Go up.")
            else:
                v_delta = 0



            new_pos = (current_pos[0] + h_delta, current_pos[1] + v_delta)
            print("    Sending:")
            print("       Pan:  %d" %(new_pos[0]))
            print("       Tilt: %d" %(new_pos[1]))

            self.motors.setPTMotorsData(new_pos[0], new_pos[1], self.limits.maxPanSpeed, self.limits.maxTiltSpeed)
