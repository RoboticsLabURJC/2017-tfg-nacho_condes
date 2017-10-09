#
# Created on Mar 12, 2017
#
# @author: dpascualhe
#

import time

import cv2
import numpy as np
from keras import backend
import matplotlib.pyplot as plt
from keras.utils import np_utils, io_utils
from keras.preprocessing import image as imkeras

class NetData:

    def __init__(self, im_rows, im_cols, nb_classes, verbose):
        ''' NetData class adapts and augments datasets. '''
        self.im_rows = im_rows
        self.im_cols = im_cols
        self.nb_classes = nb_classes
        
        self.verbose = verbose
        self.count = 0
    
    def load(self, path):
        ''' Loads a HDF5 dataset. '''
        print("\nLoading " + path + "...")
        x = np.array(io_utils.HDF5Matrix(path, "data"))
        y = np.array(io_utils.HDF5Matrix(path, "labels"))
        print("Loaded\n")
        
        return (x, y)
        
    def adapt(self, X, Y):
        ''' Adapts the dataset shape and format depending on Keras
        backend: TensorFlow or Theano.
        '''
        if backend.image_dim_ordering() == "th":
            x = X.reshape(X.shape[0], 1, self.im_rows, self.im_cols)
            input_shape = (1, self.im_rows, self.im_cols)
        else:
            x = X.reshape(X.shape[0], self.im_rows, self.im_cols, 1)
            input_shape = (self.im_rows, self.im_cols, 1)
         
        x = x.astype('float32')
        x /= 255 # Normalizes data: [0,255] -> [0,1]
            
        # Converts class vector to class matrix
        y = np_utils.to_categorical(Y, self.nb_classes)
        
        print('Original input images data shape: ', X.shape)
        print('Input images data reshaped: ', x.shape)
        print('----------------------------------------------------------')
        print('Input images type: ', X.dtype)
        print('New input images type: ', x.dtype)
        print('----------------------------------------------------------')
        print('Original class label data shape: ', Y.shape)
        print('Class label data reshaped: ', y.shape)
        print('----------------------------------------------------------\n')
        
        if self.verbose == "y":
            i = 0
            X = x.reshape(x.shape[0], 28, 28)
            for im in X:
                plot_count = 241 + i
                plt.subplot(plot_count)
                plt.imshow(im, cmap='gray')
                i += 1
                if i == 8:
                    break
            plt.show()
                
        return (x, y), input_shape

    def sobelEdges(self, sample):
        ''' Apply a sobel filtering in x and y directions in order to
        detect edges. It's used right before data enters the net.
        '''
        im_sobel_x = cv2.Sobel(sample, cv2.CV_32F, 1, 0, ksize=5)
        im_sobel_y = cv2.Sobel(sample, cv2.CV_32F, 0, 1, ksize=5)
        im_edges = cv2.add(abs(im_sobel_x), abs(im_sobel_y))
        im_edges = cv2.normalize(im_edges, None, 0, 1, cv2.NORM_MINMAX)
        im_edges = im_edges[ : , : ,np.newaxis]
        
        return im_edges

    def augmentation(self, x, y, batch_size, mode):
        ''' Creates a generator that augments data in real time. It can
        apply only a Sobel filtering or a stack of processes that
        randomize the data.
        '''      
        if mode == "full":
            datagen = imkeras.ImageDataGenerator(
                zoom_range=0.2, rotation_range=20, width_shift_range=0.2, 
                height_shift_range=0.2, fill_mode='constant', cval=0,
                preprocessing_function=self.sobelEdges)
        elif mode == "edges":
            datagen = imkeras.ImageDataGenerator(
                preprocessing_function=self.sobelEdges)
  
        generator = datagen.flow(x, y, batch_size=batch_size)
        
        if verbose == "y":
            print("Determining class distribution...")
            i = 0
            first_batch = 1
            classes_count = [0,0,0,0,0,0,0,0,0,0]
            for x_batch, y_batch in generator:                
                if first_batch:              
                    x_batch = x_batch.reshape(x_batch.shape[0], 28, 28)
                    for im in x_batch:
                        plot_count = 241 + i
                        plt.subplot(plot_count)
                        plt.imshow(im, cmap='gray')
                        i += 1
                        if i == 8:
                            break
                    first_batch = 0
                    plt.show()
                    
                if self.count == 0:
                    for classes in y_batch:
                        if np.where(classes == 1)[0] == [0]:
                            classes_count[0] += 1
                        elif np.where(classes == 1)[0] == [1]:
                            classes_count[1] += 1
                        elif np.where(classes == 1)[0] == [2]:
                            classes_count[2] += 1
                        elif np.where(classes == 1)[0] == [3]:
                            classes_count[3] += 1
                        elif np.where(classes == 1)[0] == [4]:
                            classes_count[4] += 1
                        elif np.where(classes == 1)[0] == [5]:
                            classes_count[5] += 1
                        elif np.where(classes == 1)[0] == [6]:
                            classes_count[6] += 1
                        elif np.where(classes == 1)[0] == [7]:
                            classes_count[7] += 1
                        elif np.where(classes == 1)[0] == [8]:
                            classes_count[8] += 1
                        elif np.where(classes == 1)[0] == [9]:
                            classes_count[9] += 1
                    i += 1
                    if i >= 3000:
                        print("Class distribution: ", classes_count)
                        break
                else:
                    break        
            self.count += 1
        
        return generator

    def samples_extraction(self, x, nb_samples):
        ''' Extracts a given number of samples from the given dataset
        and saves them in the Datasets/Samples/ folder
        '''
        timestr = time.strftime("%Y%m%d-%H%M%S")
        x = x.reshape(x.shape[0], 28, 28)
        for i in range(nb_samples):
            im = x[i] * 255
            cv2.imwrite("../Datasets/Samples/" + timestr + "_" + str(i) + ".png", im)
    
