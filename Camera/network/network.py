import tensorflow as tf
import numpy as np
import cv2

import h5importer
from cprint import *

from tensorflow.examples.tutorials.mnist import input_data

import random as rnd

from customevaluation import CustomEvaluation


class Network():

    def __init__(self):
        # initializing placeholders
        self.x = tf.placeholder(tf.float32, shape=[None, 784], name='x')
        self.y_ = tf.placeholder(tf.float32, shape=[None, 10], name='y_')
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')

        # common session to share variables
        self.sess = tf.Session()

        # initalizing methods.
        def weight_variable(shape, name):
            initial = tf.truncated_normal(shape, stddev=0.1)
            return tf.Variable(initial, name=name)

        def bias_variable(shape, name):
            initial = tf.constant(0.1, shape=shape)
            return tf.Variable(initial, name=name)
        
        # variables.
        self.weights = {
            'conv1': weight_variable([5, 5, 1, 32], 'conv1'),
            'conv2': weight_variable([5, 5, 32, 32], 'conv2'),
            'fc1':   weight_variable([14*14*32, 128], 'fc1'),
            'fc2':   weight_variable([128, 10], 'fc2')
        }
        self.biases = {
            'conv1': bias_variable([32], 'conv1'),
            'conv2': bias_variable([32], 'conv2'),
            'fc1':   bias_variable([128], 'fc1'),
            'fc2':   bias_variable([10], 'fc2')
        }

        self.saver = tf.train.Saver({'W_conv1': self.weights['conv1'], 'b_conv1': self.biases['conv1'],
                        'W_conv2': self.weights['conv2'], 'b_conv2': self.biases['conv2'],
                        'W_fc1': self.weights['fc1'], 'b_fc1': self.biases['fc1'],
                        'W_fc2': self.weights['fc2'], 'b_fc2': self.biases['fc2']
                        })


        self.sess.run(tf.global_variables_initializer())


        # model definition.

        self.x_image = tf.reshape(self.x, [-1, 28, 28, 1])

        self.h_conv1 = tf.nn.relu(self.conv2d(self.x_image, self.weights['conv1']) + self.biases['conv1'])
        self.h_conv2 = tf.nn.relu(self.conv2d(self.h_conv1, self.weights['conv2']) + self.biases['conv2'])
        self.h_pool = self.max_pool_2x2(self.h_conv2)
        self.h_drop1 = tf.nn.dropout(self.h_pool, self.keep_prob)
        self.h_pool_flat = tf.reshape(self.h_drop1, [-1, 14*14*32])
        self.h_fc1 = tf.nn.relu(tf.matmul(self.h_pool_flat, self.weights['fc1']) + self.biases['fc1'])
        self.h_drop2 = tf.nn.dropout(self.h_fc1, self.keep_prob)
        self.y = tf.nn.softmax(tf.nn.relu(tf.matmul(self.h_drop2, self.weights['fc2']) + self.biases['fc2']))


        self.mnist = input_data.read_data_sets('/tmp/MNIST_data', one_hot=True)

    def conv2d(self, x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
    def max_pool_2x2(self, x):
        return tf.nn.max_pool(x, ksize=[1, 2, 3, 1], strides=[1, 2, 2, 1], padding='SAME')


    def train(self, parameters):
        '''
        Trains the model, and saves the parameters into a checkpoint
        '''
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.y_, logits=self.y))
        train_step = tf.train.AdamOptimizer(parameters['learning_rate']).minimize(cross_entropy)
        accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.y, 1), tf.argmax(self.y_,1)), tf.float32))

        
        if parameters['training_dataset_path'] == '':
            training_dataset = self.mnist.train
        else:
            training_dataset = h5importer.HDF5Importer(parameters['training_dataset_path'], 'training.png')

        if parameters['validation_dataset_path'] == '':
            validation_dataset = self.mnist.validation
        else:
            validation_dataset = h5importer.HDF5Importer(parameters['validation_dataset_path'], 'validation.png')


        training_length = parameters['train_steps']

        total_training_loss = []
        total_training_accuracy = []
        total_validation_loss = []
        total_validation_accuracy = []

        previous_monitored = None
        early_stopping_counter = 0
        improvement = None


        # training process
        self.sess.run(tf.global_variables_initializer())

        for step in range(training_length):
            trainBatch = training_dataset.next_batch(parameters['batch_size'])
            #print(step)
            if (step + 1) % 100 == 0:
                validationBatch = validation_dataset.next_batch(parameters['batch_size'])
                print("step: %d" % (step+1))
                validation_loss = self.sess.run(cross_entropy, 
                                          feed_dict={self.x: validationBatch[0], 
                                                     self.y_: validationBatch[1], 
                                                     self.keep_prob: 1.0})
                total_validation_loss.append(validation_loss)


                validation_accuracy = self.sess.run(accuracy,
                                                  feed_dict={self.x: validationBatch[0],
                                                             self.y_: validationBatch[1],
                                                             self.keep_prob: 1.0})
                total_validation_accuracy.append(validation_accuracy)
                path = self.saver.save(self.sess, parameters['model_path'], global_step=step+1)


            training_loss = self.sess.run(cross_entropy, 
                                          feed_dict={self.x: trainBatch[0], 
                                                     self.y_: trainBatch[1], 
                                                     self.keep_prob: 1.0})
            total_training_loss.append(training_loss)



            training_accuracy = self.sess.run(accuracy,
                                              feed_dict={self.x: trainBatch[0],
                                                         self.y_: trainBatch[1],
                                                         self.keep_prob: 1.0})
            total_training_accuracy.append(training_accuracy)

            # early stopping implementation
            if parameters['early_stopping'] == 'y' or previous_monitored:
                if parameters['monitor'] == 'accuracy':
                    improvement = (training_accuracy > previous_monitored)
                    previous_monitored = training_accuracy
                elif parameters['monitor'] == 'loss':
                    improvement = (training_loss < previous_monitored)
                    previous_monitored = training_loss
                else:
                    print('Error on monitored parameter.')

                
                if not improvement:
                    early_stopping_counter += 1
                    #cprint.warn("Stopping in: %d" % (parameters['patience'] - early_stopping_counter))
                    if early_stopping_counter > parameters['patience']:
                        cprint.fatal("Training not improved. Process halted on step %d." % (step))
                        break
                else:
                    early_stopping_counter = 0

                

            self.sess.run(train_step, feed_dict= {self.x: trainBatch[0], 
                                                  self.y_: trainBatch[1], 
                                                  self.keep_prob: 0.5})


        print("training completed\nSaved in %s\n\n" %(path))

        return (total_training_accuracy, total_training_loss, total_validation_accuracy, total_validation_loss)


    def load(self, path):
        '''
        Loads the latest model saved on path
        '''
        latest_model = tf.train.latest_checkpoint(path)
        print(latest_model)
        self.saver.restore(self.sess, latest_model)

    def predict(self, img):
        '''
        gets an image and processes it, returning the predicted label
        '''
        img = img.reshape([1,784])
        return self.sess.run(self.y, feed_dict={self.x: img, self.keep_prob: 1.0})

    def test(self, test_parameters):
        if test_parameters['test_dataset_path'] == '':
            test_dataset = self.mnist.test
        else:
            test_dataset = h5importer.HDF5Importer(test_parameters['test_dataset_path'], 'test.png')

        testBatch = test_dataset.next_batch(test_parameters['n_samples'])

        test_output = self.sess.run(self.y, feed_dict={self.x: testBatch[0], self.keep_prob: 1.0})

        if test_parameters['is_training']:
            is_training = 'y'
            train_loss = test_parameters['train_loss']
            train_acc = test_parameters['train_acc']
            val_loss = test_parameters['val_loss']
            val_acc = test_parameters['val_acc']
        else:
            is_training = 'n'
            train_loss = None
            train_acc = None
            val_loss = None
            val_acc = None

        results = CustomEvaluation(np.argmax(testBatch[1], axis=1), test_output, is_training, train_loss, train_acc, val_loss, val_acc)

        score_dict = results.dictionary()

        results.log(score_dict, test_parameters['output_matrix'])
       