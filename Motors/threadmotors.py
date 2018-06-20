#
# Created on Jan 26, 2018
#
# @author: naxvm
#

import time
import threading
from datetime import datetime


class ThreadMotors(threading.Thread):


    def __init__(self, motors):
        ''' Threading class for Motors. '''

        self.t_cycle = 100  # ms

        self.motors = motors

        self.is_activated = True

        threading.Thread.__init__(self)


    def run(self):
        ''' Updates the thread. '''
        while(True):
            start_time = datetime.now()
            if self.is_activated:
                self.motors.move()
            else:
                try:
                    self.motors.motors.sendW(0)
                    self.motors.motors.sendVX(0)
                except AttributeError:
                    pass # when PTZ instead of turtlebot
            end_time = datetime.now()

            dt = end_time - start_time
            dtms = ((dt.days * 24 * 60 * 60 + dt.seconds) * 1000 +
                    dt.microseconds / 1000.0)


            if(dtms < self.t_cycle):
                time.sleep((self.t_cycle - dtms) / 1000.0)

    def toggle(self):
        self.is_activated = not self.is_activated

    def runOnce(self):
        if not self.is_activated:
            self.motors.move()
