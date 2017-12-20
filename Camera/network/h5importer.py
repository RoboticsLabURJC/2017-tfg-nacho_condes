import h5py
import numpy as np
import matplotlib.pyplot as plt
import random as rnd


class HDF5Importer():
    def __init__(self, filename, image_name=None):
        self.file = h5py.File(filename, 'r')

        raw_data = self.file['data']
        raw_labels = self.file['labels']

        self.data, self.labels = self.convert(raw_data, raw_labels)
        self.length = len(self.data)
        print("shape: {}".format(self.data.shape))
        if image_name:
        	self.plot_samples(np.random.randint(0, self.length, 9), image_name)


    def convert(self, data, labels):
        # labels conversion to one_hot format
        self.dataSet_size = labels.size

        arrLabels = np.zeros([self.dataSet_size, 10], dtype='float32')
        arrImgs = np.zeros([self.dataSet_size, 28, 28], dtype='float32')

        for index in range(self.dataSet_size):
            currentImg = np.true_divide(data[index], 255.0)
            currentLabel = labels[index]
            arrImgs[index] = currentImg.reshape([28,28])
            arrLabels[index][currentLabel] = 1.0

        return arrImgs, arrLabels


    def plot_samples(self, numbers, image_name):
        f, ax = plt.subplots(3,3, figsize=(15,15))
        plt.gray()
        counter = 0
        for row in ax:
            for a in row:
                a.imshow(self.data[numbers[counter]])
                a.set_title('Label: {0} (sample {1})'.format(self.labels[numbers[counter]].argmax(), numbers[counter]))
                counter += 1
        plt.show(block=False)
        plt.savefig(image_name, bbox_inches='tight')
        plt.close()


    def next_batch(self, batchSize):
        indices = np.random.randint(self.dataSet_size, size=batchSize)

        batchImgs = self.data[indices]
        batchImgs = batchImgs.reshape([batchSize, 784])

        batchLabels = self.labels[indices]

        return batchImgs, batchLabels
