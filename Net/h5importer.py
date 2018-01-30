#
# Created on Dec, 2017
#
# @author: naxvm
#

import h5py
import numpy as np
import matplotlib.pyplot as plt
import random as rnd


class HDF5Importer():
    '''Class to import a HDF5 dataset, originally saved with the data-labels
    format. Optionally, it plots on a figure nine random samples from it,
    saving that figure as an image, if desired.'''

    def __init__(self, filename, image_name=None):
        self.file = h5py.File(filename, 'r')
        # We retrieve the raw data from the file.
        raw_data = self.file['data']
        raw_labels = self.file['labels']
        # And convert it to np format.
        self.data, self.labels = self.convert(raw_data, raw_labels)
        self.length = len(self.data)
        print("shape: {}".format(self.data.shape))

        # Plotting 9 samples if required
        if image_name:
            self.plot_samples(np.random.randint(0, self.length, 9), image_name)

    def convert(self, data, labels):
        '''Labels conversion to one-hot ([0 0 1 0 0...]) format.
        Reshaping to 28x28 and scaling the data from 0-255 to 0-1 (float)'''
        self.dataSet_size = labels.size

        arrLabels = np.zeros([self.dataSet_size, 10], dtype='float32')
        arrImgs = np.zeros([self.dataSet_size, 28, 28], dtype='float32')

        for index in range(self.dataSet_size):
            currentImg = np.true_divide(data[index], 255.0)
            currentLabel = labels[index]
            arrImgs[index] = currentImg.reshape([28, 28])
            arrLabels[index][currentLabel] = 1.0

        return arrImgs, arrLabels

    def plot_samples(self, numbers, image_name):
        '''Method to generate a figure with 9 random images
        from the dataset, correspondly titled with their labels.
        It allows to save the figure as image_name.png.'''
        f, ax = plt.subplots(3, 3, figsize=(15, 15))
        plt.gray()
        counter = 0
        for row in ax:
            for a in row:
                a.imshow(self.data[numbers[counter]])
                a.set_title('Label: {0} (sample {1})'.format(
                    self.labels[numbers[counter]].argmax(), numbers[counter]))
                counter += 1
        plt.show(block=False)
        plt.savefig(image_name, bbox_inches='tight')
        plt.close()

    def next_batch(self, batchSize):
        '''Method to iterate over the dataset. It returns a random
        batch of images-labels from the dataset
        (containing batchSize images).'''
        indices = np.random.randint(self.dataSet_size, size=batchSize)

        batchImgs = self.data[indices]
        # Reshaping to the CNN format (28x28 -> 1x784)
        batchImgs = batchImgs.reshape([batchSize, 784])
        
        batchLabels = self.labels[indices]

        return batchImgs, batchLabels
