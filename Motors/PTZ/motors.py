from cprint import *
class Motors():

    def __init__(self, motors):
        self.motors = motors
        self.limits = self.motors.getLimits()
        self.initial = True

        self.last_center = (0, 0)
        self.threshold = 60

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
        current_pos = self.motors.motors.data
        current_pos = (current_pos.pan, current_pos.tilt)

        if index is not None:
            cprint.info('---Person detected---')

            horz_error = box_center[0] - self.center_coords[0]
            vert_error = box_center[1] - self.center_coords[1]
            # Horizontal delta:
            if abs(horz_error) > self.threshold:
                if box_center[0] > self.center_coords[0]:
                    cprint.warn('  Horizontal distance: %d. Going right.' % (horz_error))
                    h_delta = 2
                else:
                    cprint.warn('  Horizontal distance: %d. Going left.' % (horz_error))
                    h_delta = -2
            else:
                cprint.ok('  Horizontal distance: %d (under control).' % (horz_error))
                h_delta = 0

            # Vertical delta
            if abs(vert_error) > self.threshold:
                if box_center[1] > self.center_coords[1]:
                    v_delta = -2
                    cprint.warn('  Vertical distance: %d. Going down.' % (vert_error))
                else:
                    v_delta = 2
                    cprint.warn('  Vertical distance: %d. Going up.' % (vert_error))
            else:
                cprint.ok('  Vertical distance: %d (under control).' % (vert_error))
                v_delta = 0

            new_pos = (current_pos[0] + h_delta, current_pos[1] + v_delta)

            self.motors.setPTMotorsData(new_pos[0], new_pos[1], self.limits.maxPanSpeed, self.limits.maxTiltSpeed)
            self.last_center = box_center
            self.initial = False

        elif not self.initial:
            last_x = self.last_center[0]

            h_loss = self.center_coords[0] - last_x
            if h_loss < 0:
                h_delta = 1
                cprint.fatal('  Person lost. Turning right.')
            else:
                h_delta = -1
                cprint.fatal('  Person lost. Turning left.')

            self.motors.setPTMotorsData(current_pos[0] + h_delta, current_pos[1], self.limits.maxPanSpeed, self.limits.maxTiltSpeed)

        else: # there are no people in the image
            cprint.warn('---Nothing detected yet...---')
