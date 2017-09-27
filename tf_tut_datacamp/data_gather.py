#! /usr/bin/python2

from skimage import data
import os
import tensorflow as tf
import numpy as np

def data_gather():
    def load_data(data_directory):
        directories = [d for d in os.listdir(data_directory) 
                        if os.path.isdir(os.path.join(data_directory, d))] # gets an array of directories inside the root
        labels = []
        images = []
        for d in directories: # iterate over each dir
            label_directory = os.path.join(data_directory, d)
            file_names = [os.path.join(label_directory, f)
                            for f in os.listdir(label_directory)
                            if f.endswith(".ppm")]
            for f in file_names:
                images.append(data.imread(f))
                labels.append(int(d))
        return images, labels

    ROOT_PATH = "/home/nacho/tens/tf_tut_datacamp"
    train_data_directory = os.path.join(ROOT_PATH, "TrafficSigns/Training")
    test_data_directory = os.path.join(ROOT_PATH, "TrafficSigns/Testing")

    images, labels = load_data(train_data_directory)
    return images, labels
