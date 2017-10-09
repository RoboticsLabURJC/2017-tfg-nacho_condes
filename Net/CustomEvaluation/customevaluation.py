#
# Created on Mar 12, 2017
#
# @author: dpascualhe
#

import numpy as np
import scipy.io as sio
from sklearn import metrics

class CustomEvaluation():
    def __init__(self, y_test, y_proba=[], training="n", train_loss=[],
                 train_acc=[], val_loss=[], val_acc=[]):
        """ CustomEvaluation class outputs a dictionary with a variety of
        metrics to evaluate the neural network performance.
        """
        # Test labels.
        self.y_test = y_test
        self.y_proba = y_proba
        self.y_pred = np.argmax(self.y_proba, axis=1)
        
        # Training metrics.
        self.train_loss = train_loss
        self.train_acc = train_acc
        self.val_loss = val_loss
        self.val_acc = val_acc
        self.training = training

    def dictionary(self):
        """ Calculates test measures and saves them into a dictionary
        beside the training ones.
        """
        # Test metrics.
        conf_mat = metrics.confusion_matrix(self.y_test, self.y_pred)
        loss = metrics.log_loss(self.y_test, self.y_proba, 10e-8)
        acc = metrics.accuracy_score(self.y_test, self.y_pred)
        pre = metrics.precision_score(self.y_test, self.y_pred, average=None)    
        rec = metrics.recall_score(self.y_test, self.y_pred, average=None)
    
        # We save the metrics into a dictionary.
        measures_dict = {"confusion matrix": conf_mat, "loss": loss, 
                        "accuracy": acc, "precision": pre, "recall": rec,
                        "training": self.training}
        
        if self.training == "y":
            measures_dict["training loss"] = self.train_loss
            measures_dict["training accuracy"] = self.train_acc
            measures_dict["validation loss"] = self.val_loss
            measures_dict["validation accuracy"] = self.val_acc

        return measures_dict

    def log(self, measures_dict):
        """ Logs the results into a .mat file for Octave. """
        sio.savemat("measures.mat", {"metrics": measures_dict})
