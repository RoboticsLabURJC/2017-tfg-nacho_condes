import tensorflow as tf
import numpy as np
import cv2
import progressbar as pb
import os.path

import h5importer
from cprint import *

from tensorflow.examples.tutorials.mnist import input_data

import random as rnd

from customevaluation import CustomEvaluation


class Network():

    def __init__(self):
        # initializing placeholders
        self.x = tf.placeholder(tf.float32, shape=[None, 784], name='x')

        self.x_image = tf.reshape(self.x, [-1, 28, 28, 1])

        self.y_ = tf.placeholder(tf.float32, shape=[None, 10], name='y_')
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')

        # common session to share variables
        self.sess = tf.Session()

        # initializing methods.
        def weight_variable(shape, name):
            initial = tf.truncated_normal(shape, stddev=0.1)
            return tf.Variable(initial, name=name)

        def bias_variable(shape, name):
            initial = tf.constant(0.1, shape=shape)
            return tf.Variable(initial, name=name)

        # variables.
        self.weights = {}
        self.biases = {}

        # conv1
        with tf.name_scope('conv1'):
            self.weights['conv1'] = weight_variable([5, 5, 1, 32], 'W_conv1')
            self.biases['conv1'] = bias_variable([32], 'b_conv1')
            self.h_conv1 = tf.nn.relu(self.conv2d(self.x_image, self.weights['conv1']) + self.biases['conv1'])

        # conv2
        with tf.name_scope('conv2'):
            self.weights['conv2'] = weight_variable([5, 5, 32, 32], 'W_conv2')
            self.biases['conv2'] = bias_variable([32], 'b_conv2')
            self.h_conv2 = tf.nn.relu(self.conv2d(self.h_conv1, self.weights['conv2']) + self.biases['conv2'])

        self.h_pool = self.max_pool_2x2(self.h_conv2)
        self.h_drop1 = tf.nn.dropout(self.h_pool, self.keep_prob)
        self.h_pool_flat = tf.reshape(self.h_drop1, [-1, 14*14*32])

        # fc1
        with tf.name_scope('fc1'):
            self.weights['fc1'] = weight_variable([14*14*32, 128], 'W_fc1')
            self.biases['fc1'] = bias_variable([128], 'b_fc1')
            self.h_fc1 = tf.nn.relu(tf.matmul(self.h_pool_flat, self.weights['fc1']) + self.biases['fc1'])

        self.h_drop2 = tf.nn.dropout(self.h_fc1, self.keep_prob)

        # fc2
        with tf.name_scope('fc2'):
            self.weights['fc2'] = weight_variable([128, 10], 'W_fc2')
            self.biases['fc2'] = bias_variable([10], 'b_fc2')
            self.y = tf.nn.softmax(tf.nn.relu(tf.matmul(self.h_drop2, self.weights['fc2']) + self.biases['fc2']))

        self.saver = tf.train.Saver({'W_conv1': self.weights['conv1'], 'b_conv1': self.biases['conv1'],
                        'W_conv2': self.weights['conv2'], 'b_conv2': self.biases['conv2'],
                        'W_fc1': self.weights['fc1'], 'b_fc1': self.biases['fc1'],
                        'W_fc2': self.weights['fc2'], 'b_fc2': self.biases['fc2']
                        })

        self.sess.run(tf.global_variables_initializer())

        # model definition.

        self.mnist = input_data.read_data_sets('/tmp/MNIST_data', one_hot=True)

    def conv2d(self, x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
    def max_pool_2x2(self, x):
        return tf.nn.max_pool(x, ksize=[1, 2, 3, 1], strides=[1, 2, 2, 1], padding='SAME')


    def train(self, parameters):
        '''
        Trains the model, and saves the parameters into a checkpoint
        '''
        writer = tf.summary.FileWriter((parameters['model_path'] + '/logs'), graph=tf.get_default_graph())

        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.y_, logits=self.y))
        train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
        accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.y, 1), tf.argmax(self.y_,1)), tf.float32))
        
        if parameters['training_dataset_path'] == '':
            training_dataset = self.mnist.train
        else:
            training_dataset = h5importer.HDF5Importer(parameters['training_dataset_path'], 'training.png')

        if parameters['validation_dataset_path'] == '':
            validation_dataset = self.mnist.validation
        else:
            validation_dataset = h5importer.HDF5Importer(parameters['validation_dataset_path'], 'validation.png')


        total_training_loss = []
        total_training_accuracy = []
        total_validation_loss = []
        total_validation_accuracy = []

        if parameters['monitor'] == 'accuracy':
            previous_monitored = 0
        else:
            previous_monitored = 99999

        early_stopping_counter = 0


        # training process
        self.sess.run(tf.global_variables_initializer())

        DATASET_SIZE = 48000
        BATCH_SIZE = 128
        batch_count = DATASET_SIZE / BATCH_SIZE # 375
        TOTAL_EPOCHS = 100

        VALIDATION_SIZE = 2000

        batch = 0

        for epoch in range(TOTAL_EPOCHS):       # each epoch must iterate over all the dataset -> 25 times over all the dataset (forwards only for the moment).
            print("Epoch %d/%d" % (epoch+1, TOTAL_EPOCHS))
            bar = pb.ProgressBar(max_value=batch_count,
                                 redirect_stdout=True,
                                 widgets=[
                                    ' [', pb.Timer(), '] ',
                                    pb.Bar(), ' (', pb.ETA(), ') '])


            for batch in range(batch_count):
                train_batch = training_dataset.next_batch(BATCH_SIZE)


                training_loss = self.sess.run(cross_entropy, feed_dict={self.x: train_batch[0],
                                                                        self.y_: train_batch[1],
                                                                        self.keep_prob: 1.0})

                training_accuracy = self.sess.run(accuracy, feed_dict={self.x: train_batch[0],
                                                                       self.y_: train_batch[1],
                                                                       self.keep_prob: 1.0})

                total_training_loss.append(training_loss)
                total_training_accuracy.append(training_accuracy)


                # print("batch %d, loss: %.3f, acc: %.3f" % (batch, training_loss, training_accuracy))
            
                self.sess.run(train_step, feed_dict= {self.x: train_batch[0],
                                                  self.y_: train_batch[1],
                                                  self.keep_prob: 0.5})
                bar.update(batch+1)


            # for each epoch, we validate the model
            validation_batch = validation_dataset.next_batch(VALIDATION_SIZE)

            validation_loss = self.sess.run(cross_entropy, feed_dict={self.x: validation_batch[0],
                                                                      self.y_: validation_batch[1],
                                                                      self.keep_prob: 1.0})


            validation_accuracy = self.sess.run(accuracy, feed_dict={self.x: validation_batch[0],
                                                                     self.y_: validation_batch[1],
                                                                     self.keep_prob: 1.0})

            total_validation_loss.append(validation_loss)
            total_validation_accuracy.append(validation_accuracy)

            print("VALIDATION EPOCH %d: loss %.3f, accuracy %.3f" %(epoch+1, validation_loss, validation_accuracy))

            if parameters['early_stopping']:
                if parameters['monitor'] == 'accuracy':
                    if validation_accuracy > previous_monitored:
                        path = self.saver.save(self.sess, (parameters['model_path'] + '/model'), global_step=epoch+1)
                        cprint.ok('Accuracy improved from {0} to {1}. Saving at {2}'.format(previous_monitored, validation_accuracy, path))
                        previous_monitored = validation_accuracy
                    else:
                        cprint.warn('Accuracy did not improve.')
                        early_stopping_counter += 1
                else:
                    if validation_loss < previous_monitored:
                        path = self.saver.save(self.sess, (parameters['model_path'] + '/model'), global_step=epoch+1)
                        cprint.ok('Loss improved from {0} to {1}. Saving at {2}'.format(previous_monitored, validation_loss, path))
                        previous_monitored = validation_loss
                    else:
                        cprint.warn('Loss did not improve.')
                        early_stopping_counter += 1
                if early_stopping_counter > parameters['patience']:
                    cprint.fatal('Patience exceeded. Training halted. Best model saved on {}'.format(path))
                    break

        '''
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
                path = self.saver.save(self.sess, (parameters['model_path'] + '/model'), global_step=step+1)


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
        '''

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

        test_batch = test_dataset.next_batch(2000)

        test_output = self.sess.run(self.y, feed_dict={self.x: test_batch[0], self.keep_prob: 1.0})

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

        results = CustomEvaluation(np.argmax(test_batch[1], axis=1), test_output, is_training, train_loss, train_acc, val_loss, val_acc)

        score_dict = results.dictionary()

        results.log(score_dict, test_parameters['output_matrix'])


#
# MANAGER METHODS
#
def do_train(final_test=False):

    net = Network()

    model_path = raw_input('Enter the path to save the network (leave blank for "my-model"): ')
    if model_path == '':
        model_path = 'my-model'

    training_dataset_path = raw_input('Enter the path to the training dataset .h5 file (leave blank to use standard MNIST): ')

    while training_dataset_path != '' and not os.path.isfile(training_dataset_path):
        training_dataset_path = raw_input('    Please enter a correct path to the .h5 file (leave blank to use standard MNIST): ')

    validation_dataset_path = raw_input('Enter the path to the validation dataset .h5 file (leave blank to use standard MNIST): ')
    while validation_dataset_path != '' and not os.path.isfile(validation_dataset_path):
        validation_dataset_path = raw_input('    Please enter a correct path to the .h5 file (leave blank to use standard MNIST): ')

    early_stopping = raw_input('Do you want to implement early stopping (y/n)? ')
    while early_stopping != 'y' and early_stopping != 'n':
        early_stopping = raw_input('    Please enter y (yes) or n (no): ')

    monitor = None
    patience =  None
    if early_stopping == 'y':
        monitor = raw_input('    What do you want to monitor (accuracy/loss)? ')
        while monitor != 'accuracy' and monitor != 'loss':
            monitor = raw_input('    Please enter "accuracy" or "loss": ')
        patience = raw_input('    Enter patience (leave blank for 2): ')
        if patience == '':
            patience = 2
        else:
            patience = int(patience)


    parameters = {'model_path': model_path,
                  'training_dataset_path': training_dataset_path,
                  'validation_dataset_path': validation_dataset_path,
                  'early_stopping': early_stopping,
                  'monitor': monitor,
                  'patience': patience}
    (train_acc, train_loss, val_acc, val_loss) = net.train(parameters)

    if final_test:
        test_parameters = get_test_parameters(is_training=True)
        test_parameters['train_acc'] = train_acc
        test_parameters['train_loss'] = train_loss
        test_parameters['val_acc'] = val_acc
        test_parameters['val_loss'] = val_loss
        net.test(test_parameters)


def get_test_parameters(is_training=False):
    test_dataset_path = raw_input('Enter the path to the testing dataset .h5 file (leave blank to use standard MNIST): ')
    while test_dataset_path != '' and not os.path.isfile(test_dataset_path):
        test_dataset_path = raw_input('    Please enter a correct path to the .h5 file (leave blank to use standard MNIST): ')

    output_matrix = raw_input('Enter the desired name for the output matrix .mat file (leave blank to "results"): ')
    if output_matrix == '':
        output_matrix = 'results'
        
    output_matrix = (output_matrix + '.mat')

    test_parameters = {'test_dataset_path': test_dataset_path,
                       'is_training': is_training,
                       'output_matrix': output_matrix}

    return test_parameters






if __name__ == '__main__':
    action = None
    while action != 'train' and action != 'test' and action != 'both':
        action = raw_input('\nWhat do you want to do (train/test/both)? ')

    if action == 'train':
        do_train()

    elif action == 'test':
        net = Network()
        model_path = raw_input('Enter the path containing the model to evaluate (leave blank for "my-model"): ')
        while model_path != '' and not os.path.exists(model_path):
            model_path = raw_input('    Please enter a valid path (leave blank for "my-model")')
        if model_path == '':
            model_path = 'my-model'

        net.load(model_path)

        test_parameters = get_test_parameters()
        net.test(test_parameters)

    else:
        do_train(final_test=True)
