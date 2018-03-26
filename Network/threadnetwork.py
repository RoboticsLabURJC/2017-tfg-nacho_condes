#
# Created on Feb 19, 2018
#
__author__ = "Naxvm"

import time
import threading
from datetime import datetime


class ThreadNetwork(threading.Thread):

    def __init__(self, network):
        ''' Threading class for Network.'''

        self.t_cycle = 100  # ms

        self.network = network
        threading.Thread.__init__(self)
        self.is_activated = True
        self.framerate = 0

        self.lock = threading.Lock()

    def run(self):
        ''' Updates the thread.'''
        while(True):
            start_time = datetime.now()
            if self.is_activated:
                self.network.predict()
            end_time = datetime.now()

            dt = end_time - start_time
            dtms = ((dt.days * 24 * 60 * 60 + dt.seconds) * 1000 +
                     dt.microseconds / 1000.0)
            if self.is_activated:
                delta = max(self.t_cycle, dtms)
                self.framerate = int(1000.0 / delta)
            else:
                self.framerate = 0



            if(dtms < self.t_cycle):
                time.sleep((self.t_cycle - dtms) / 1000.0)

    def toggle(self):
        self.is_activated = not self.is_activated

    def runOnce(self):
        ''' Processes one image, and then stops again.'''
        if not self.is_activated:
            self.network.predict()
