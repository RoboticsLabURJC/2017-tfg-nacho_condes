#
# Created on Mar 20, 2017
#
# @author: dpascualhe
#

import keras

class LearningCurves(keras.callbacks.Callback):
    ''' LearningCurve class is a callback for Keras that saves accuracy
    and loss after each batch.
    '''
    
    def on_train_begin(self, logs={}):
        self.loss = []
        self.accuracy = []

    def on_batch_end(self, batch, logs={}):
        self.loss.append(float(logs.get('loss')))
        self.accuracy.append(float(logs.get('acc')))