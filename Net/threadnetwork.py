#
# Created on Feb 19, 2018
#
__author__ = "Naxvm"

import time
import threading
from datetime import datetime

t_cycle = 150  # ms


class ThreadNetwork(threading.Thread):

	def __init__(self, network):
		''' Threading class for Camera.'''
		self.network = network
		threading.Thread.__init__(self)
		self.activated = False

		self.lock = threading.Lock()

	def doUpdate(self, img):
		self.lock.acquire()
		self.network.input_image = img
		self.network.transformImage()
		self.lock.release()


	def updateImage(self, img):
		if self.activated:
			self.doUpdate(img)


	def getProcessedImage(self):
		return self.network.processed_image


	def run(self):
		''' Updates the thread.'''
		while(True):
			start_time = datetime.now()
			if self.activated:
				self.network.update()
			end_time = datetime.now()

			dt = end_time - start_time
			dtms = ((dt.days * 24 * 60 * 60 + dt.seconds) * 1000 +
					 dt.microseconds / 1000.0)

			if(dtms < t_cycle):
				time.sleep((t_cycle - dtms) / 1000.0)

	def runOnce(self, img):
		''' Processes one image, and then stops again.'''
		if not self.activated:
			self.doUpdate(img)
			self.network.update()
