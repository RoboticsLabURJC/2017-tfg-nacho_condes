#
# Original code by @naxvm
# Available on:
# 
#
import tensorflow as tf
import numpy as np
import progressbar as pb
import os.path
import h5importer
from cprint import cprint
from tensorflow.examples.tutorials.mnist import input_data
from customevaluation import CustomEvaluation
import threading
import cv2

class Network:
    '''Class which creates a CNN, specially prepared to process 28x28 images,
    typically from MNIST databases. It is capable to train a model, load a
    previously trained model, test itself, and be used to predict an input
    image. If this file is executed, it offers a network manager, which
    allows you to train and/or test a network, optionally using a saved
    model, and HDF5 datasets, or standard TensorFlow MNIST libraries.

    When a model is trained, the training logs are automatically saved too,
    so you can watch your network structure using TensorBoard.

    Args:
        [model] (str): path to a saved checkpoint to be loaded into the
               instantiated network.
    Attributes:
        x (np.array): placeholder for an input image/batch (1, 784, N)
        to be introduced to the network.

        y_ (np.array): placeholder for the ground truth labels, corresponding
        to the image(s). It is used as an input when we are testing/training
        the model.

        keep_prob (float32): placeholder for the keep_prob, it means,
        the probability (over 1.0) of a node of staying switched on
        during the call to the network. Tipycally 0.5 for training, and
        1.0 for testing/predicting.
        '''

    def __init__(self, model=None):
        # initializing placeholders
        self.x = tf.placeholder(tf.float32, shape=[None, 784], name='x')
        self.x_image = tf.reshape(self.x, [-1, 28, 28, 1])
        self.y_ = tf.placeholder(tf.float32, shape=[None, 10], name='y_')
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')

        # common session to share variables.
        self.sess = tf.Session()

        # initializing methods.
        def weight_variable(shape, name):
            initial = tf.truncated_normal(shape, stddev=0.1)
            return tf.Variable(initial, name=name)

        def bias_variable(shape, name):
            initial = tf.constant(0.1, shape=shape)
            return tf.Variable(initial, name=name)

        # Network trainable variables. Stored in dictionaries for
        # keeping it tidy.
        self.weights = {}
        self.biases = {}

        # conv1 layer
        with tf.name_scope('conv1'):
            self.weights['conv1'] = weight_variable([5, 5, 1, 32], 'W_conv1')
            self.biases['conv1'] = bias_variable([32], 'b_conv1')
            self.h_conv1 = tf.nn.relu(self.conv2d(
                self.x_image, self.weights['conv1']) + self.biases['conv1'])

        # conv2 layer
        with tf.name_scope('conv2'):
            self.weights['conv2'] = weight_variable([5, 5, 32, 32], 'W_conv2')
            self.biases['conv2'] = bias_variable([32], 'b_conv2')
            self.h_conv2 = tf.nn.relu(self.conv2d(
                self.h_conv1, self.weights['conv2']) + self.biases['conv2'])

        # max 2x2 pooling layer (sampling the activation maps).
        self.h_pool = self.max_pool_2x2(self.h_conv2)

        # first dropout layer (blackout for random nodes, to
        # firm up the model).
        self.h_drop1 = tf.nn.dropout(self.h_pool, self.keep_prob)

        # flatten layer
        self.h_pool_flat = tf.reshape(self.h_drop1, [-1, 14*14*32])

        # fc1 (first fully connected layer).
        with tf.name_scope('fc1'):
            self.weights['fc1'] = weight_variable([14*14*32, 128], 'W_fc1')
            self.biases['fc1'] = bias_variable([128], 'b_fc1')
            self.h_fc1 = tf.nn.relu(
                tf.matmul(self.h_pool_flat, self.weights['fc1']) +
                self.biases['fc1'])

        # second dropout layer.
        self.h_drop2 = tf.nn.dropout(self.h_fc1, self.keep_prob)

        # fc2 (second fully connected layer). Also, readout layer, as it
        # yields a 10-sized softmaxed output, corresponding to the
        # probability of each processed image to belong to each of
        # the 0-9 classes. The probabilities sum to 1.
        with tf.name_scope('fc2'):
            self.weights['fc2'] = weight_variable([128, 10], 'W_fc2')
            self.biases['fc2'] = bias_variable([10], 'b_fc2')
            self.y = tf.nn.softmax(tf.nn.relu(tf.matmul(
                self.h_drop2, self.weights['fc2']) + self.biases['fc2']))

        # attribute to save/load all variables to/from a checkpoint.
        self.saver = tf.train.Saver({'W_conv1': self.weights['conv1'],
                                     'b_conv1': self.biases['conv1'],
                                     'W_conv2': self.weights['conv2'],
                                     'b_conv2': self.biases['conv2'],
                                     'W_fc1': self.weights['fc1'],
                                     'b_fc1': self.biases['fc1'],
                                     'W_fc2': self.weights['fc2'],
                                     'b_fc2': self.biases['fc2']})

        # variables initialization.
        self.sess.run(tf.global_variables_initializer())

        # placeholder for the tf mnist dataset, in case we select it
        self.mnist = None

        # if a model is given as an argument, we load it now.
        if model:
            self.load(model)


            # attributes containing the input image and the output category.
        self.input_image = np.zeros([28, 28])
        self.output_digit = None
        self.processed_image = np.zeros([28, 28])

        self.lock = threading.Lock()

    def conv2d(self, x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1],
                            padding='SAME')

    def max_pool_2x2(self, x):

        return tf.nn.max_pool(x, ksize=[1, 2, 3, 1],
                              strides=[1, 2, 2, 1],
                              padding='SAME')

    def train(self, model_path, training_dataset_path,
              validation_dataset_path, early_stopping,
              monitor, patience):
        '''
        Method to train the model with HDF5 datasets or standard MNIST
        libraries from TensorFlow. It also stores the loss/accuracy values
        along the training on a matrix (.mat) file, for a subsequent
        display of their evolution with the Octave "benchmark" script.

        It also allows to implement early stopping, which means to stop
        the training if the selected monitored parameter does not improve
        on "patience" epochs.

        The arguments are filled up by the network manager,
        which gently asks the user for them when this file
        is executed (python network.py):

            model_path (str): path to save the resulting model.

            training_dataset_path (str): path to a HDF5 dataset, used for
                                         training.

            validation_dataset_path (str): path to a HDF5 dataset, used for
                                    validation (during the training).

            early_stopping (bool): flag to activate/deactivate early stopping
                                   during the training.

            monitor (str): variable (accuracy/loss) to be monitored during the
                           process by the early stopping.

            patience (int): the number of epochs that the early stopping will
                            wait without an improvement before halting the
                            training.


        This method returns the arrays containing the training/validation
        loss/accuracy, for being used later by the network manager on the
        Octave script.
        '''
        # Summary writer (for TensorBoard)
        writer = tf.summary.FileWriter((model_path + '/logs'),
                                       graph=tf.get_default_graph())

        # Loss function
        cross_entropy = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.y_,
                                                       logits=self.y))

        # Optimizer
        train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

        # Compute accuracy from predicted labels and real labels
        accuracy = tf.reduce_mean(tf.cast(
            tf.equal(tf.argmax(self.y, 1), tf.argmax(self.y_, 1)),
            tf.float32))

        # Loading the datasets (standard MNIST if empty)
        if training_dataset_path == '':
            if not self.mnist:
                self.mnist = input_data.read_data_sets(
                    '/tmp/MNIST_data', one_hot=True)
            training_dataset = self.mnist.train
        else:
            training_dataset = h5importer.HDF5Importer(
                training_dataset_path, 'training.png')

        if validation_dataset_path == '':
            if not self.mnist:
                self.mnist = input_data.read_data_sets(
                    '/tmp/MNIST_data', one_hot=True)
            validation_dataset = self.mnist.validation
        else:
            validation_dataset = h5importer.HDF5Importer(
                validation_dataset_path, 'validation.png')

        # Initializing arrays to store accuracy and loss along the epochs.
        total_training_loss = []
        total_training_accuracy = []
        total_validation_loss = []
        total_validation_accuracy = []

        # Early stopping initializing
        if monitor == 'accuracy':
            previous_monitored = 0
        else:
            previous_monitored = 99999

        early_stopping_counter = 0

        #
        # Training process:
        #
        self.sess.run(tf.global_variables_initializer())

        DATASET_SIZE = 48000
        BATCH_SIZE = 30
        batch_count = DATASET_SIZE / BATCH_SIZE  # 1600
        TOTAL_EPOCHS = 3

        VALIDATION_SIZE = 2000

        batch = 0

        cprint.cprint.ok(('BEGINNING TRAINING PROCESS THROUGH %d EPOCHS, ' +
                   'BATCH SIZE: %d') % (TOTAL_EPOCHS, BATCH_SIZE))
        cprint.cprint.ok(('--------------------------------------------------' +
                  '--------------'))

        for epoch in range(TOTAL_EPOCHS):
            # each epoch must iterate over all the dataset:
            # TOTAL_EPOCHS times over all the dataset.
            print("Epoch %d/%d" % (epoch+1, TOTAL_EPOCHS))
            bar = pb.ProgressBar(max_value=batch_count,
                                 redirect_stdout=False,
                                 widgets=[
                                    ' [', pb.Timer(), '] ',
                                    pb.Bar(), ' (', pb.ETA(), ') '])

            for batch in range(batch_count):
                # We pass each batch batch_count times through the network
                train_batch = training_dataset.next_batch(BATCH_SIZE)

                training_loss = self.sess.run(cross_entropy,
                                              feed_dict={
                                                self.x: train_batch[0],
                                                self.y_: train_batch[1],
                                                self.keep_prob: 1.0})

                training_accuracy = self.sess.run(accuracy,
                                                  feed_dict={
                                                    self.x: train_batch[0],
                                                    self.y_: train_batch[1],
                                                    self.keep_prob: 1.0})

                total_training_loss.append(training_loss)
                total_training_accuracy.append(training_accuracy)

                self.sess.run(train_step, feed_dict={self.x: train_batch[0],
                                                     self.y_: train_batch[1],
                                                     self.keep_prob: 0.5})
                bar.update(batch+1)

            # for each epoch, we validate the model
            validation_batch = validation_dataset.next_batch(VALIDATION_SIZE)

            validation_loss = self.sess.run(cross_entropy,
                                            feed_dict={
                                                self.x: validation_batch[0],
                                                self.y_: validation_batch[1],
                                                self.keep_prob: 1.0})

            validation_accuracy = self.sess.run(
                accuracy, feed_dict={
                            self.x: validation_batch[0],
                            self.y_: validation_batch[1],
                            self.keep_prob: 1.0})

            total_validation_loss.append(validation_loss)
            total_validation_accuracy.append(validation_accuracy)

            print("VALIDATION EPOCH %d: loss %.3f, accuracy %.3f" %
                  (epoch+1, validation_loss, validation_accuracy))

            if early_stopping:
                if monitor == 'accuracy':
                    if validation_accuracy > previous_monitored:
                        path = self.saver.save(self.sess,
                                               (model_path + '/model'),
                                               global_step=epoch+1)
                        cprint.cprint.ok(('Accuracy improved from {0} to {1}. ' +
                                  'Saving at {2}').format(
                                    previous_monitored, validation_accuracy,
                                    path))
                        previous_monitored = validation_accuracy
                        early_stopping_counter = 0
                    else:
                        cprint.cprint.warn('Accuracy did not improve.')
                        early_stopping_counter += 1
                else:
                    if validation_loss < previous_monitored:
                        path = self.saver.save(self.sess,
                                               (model_path + '/model'),
                                               global_step=epoch+1)
                        cprint.cprint.ok(('Loss improved from {0} to {1}. ' +
                                   'Saving at {2}').format(
                                        previous_monitored, validation_loss,
                                        path))
                        previous_monitored = validation_loss
                        early_stopping_counter = 0
                    else:
                        cprint.cprint.warn('Loss did not improve.')
                        early_stopping_counter += 1
                if early_stopping_counter > patience:
                    cprint.cprint.fatal(('Patience exceeded. Training halted. ' +
                                  'Best model saved on {}').format(path))
                    break
            else:
                path = self.saver.save(self.sess, (model_path + '/model'),
                                       global_step=epoch+1)

        print('\n\n\n\n')
        print("Training completed\nSaved in %s\n\n" % (path))

        return (total_training_accuracy, total_training_loss,
                total_validation_accuracy, total_validation_loss
                )

    def load(self, path):
        '''
        Loads the latest model saved on path.

        Arguments:
            path (str): path to the directory containing the model to be
                        loaded.
        '''
        latest_model = tf.train.latest_checkpoint(path)
        print(latest_model)
        self.saver.restore(self.sess, latest_model)

    def classify(self, img):
        '''
        Processes the given image img, returning the predicted label.
        Arguments:
            img (np.array): image to be classified
        '''
        img = img.reshape([1, 784])
        output = self.sess.run(self.y, feed_dict={
            self.x: img, self.keep_prob: 1.0})

        return output

    def transformImage(self):
        ''' Transforms the image into a 28x28 pixel grayscale image and
        applies a sobel filter (both x and y directions).
        '''
        im_crop = np.copy(self.input_image[140:340, 220:420])
        im_gray = cv2.cvtColor(im_crop, cv2.COLOR_BGR2GRAY)
        im_blur = cv2.GaussianBlur(im_gray, (5, 5), 0)  # Noise reduction.

        im_res = cv2.resize(im_blur, (28, 28))

        # Edge extraction.
        im_sobel_x = cv2.Sobel(im_res, cv2.CV_32F, 1, 0, ksize=5)
        im_sobel_y = cv2.Sobel(im_res, cv2.CV_32F, 0, 1, ksize=5)
        im_edges = cv2.add(abs(im_sobel_x), abs(im_sobel_y))
        im_edges = cv2.normalize(im_edges, None, 0, 255, cv2.NORM_MINMAX)
        im_edges = np.uint8(im_edges)
        self.processed_image = im_edges

    def update(self):
        self.output_digit = np.argmax(self.classify(self.processed_image))



    def test(self, test_dataset_path, output_matrix,
             is_training=False, train_acc=None, train_loss=None,
             val_acc=None, val_loss=None):
        ''' Tests the network (it requires a previously loaded/trained
        model on it), and saves the results on a .mat file.
        Arguments:
            test_dataset_path (str): path to the HDF5 dataset,
                                     used for testing.
            output_matrix (str): desired name for the output matrix
                                 containing the results.
            is_training (bool): flag to indicate that the test is done
                                just at the end of a training.
            [train_acc (array)]: optional vector containing the
                                 training accuracy.
            [train_loss (array)]: optional vector containing the
                                  training loss.
            [val_acc (array)]: optional vector containing the
                               validation accuracy.
            [val_loss (array)]: optional vector containing the
                                validation loss.
            '''
        print("ENTERING TO TEST")
        if test_dataset_path == '':
            if not self.mnist:
                self.mnist = input_data.read_data_sets(
                    '/tmp/MNIST_data', one_hot=True)
            test_dataset = self.mnist.test
        else:
            test_dataset = h5importer.HDF5Importer(
                test_dataset_path, 'test.png')

        test_batch = test_dataset.next_batch(2000)

        test_output = self.sess.run(
            self.y, feed_dict={
                self.x: test_batch[0], self.keep_prob: 1.0})

        if is_training:
            is_training = 'y'
        else:
            is_training = 'n'

        results = CustomEvaluation(np.argmax(test_batch[1], axis=1),
                                   test_output, is_training, train_loss,
                                   train_acc, val_loss, val_acc)
        score_dict = results.dictionary()

        results.log(score_dict, output_matrix)

#
#
# MANAGER METHODS
#
#


def do_train(final_test=False):
    '''Method which asks the user for the parameters to perform a train,
    and launches it.

    Arguments:
        [final_test (bool)]: flag that indicates if a final test is desired.
    '''

    net = Network()

    model_path = raw_input(
        'Enter the path to save the network (leave blank for "my-model"): ')
    if model_path == '':
        model_path = 'my-model'

    training_dataset_path = raw_input(
        ('Enter the path to the training dataset .h5 file ' +
         '(leave blank to use standard MNIST): '))
    while (training_dataset_path != '' and
           not os.path.isfile(training_dataset_path)):
        training_dataset_path = raw_input(
            ('    Please enter a correct path to the .h5 file ' +
             '(leave blank to use standard MNIST): '))

    validation_dataset_path = raw_input((
        'Enter the path to the validation dataset .h5 file ' +
        '(leave blank to use standard MNIST): '))
    while (validation_dataset_path != '' and
           not os.path.isfile(validation_dataset_path)):
        validation_dataset_path = raw_input(('    Please enter a correct ' +
                                             'path to the .h5 file (leave ' +
                                             'blank to use standard MNIST): ')
                                            )

    early_stopping = raw_input(('Do you want to implement early stopping ' +
                                '(y/n)? '))
    while early_stopping != 'y' and early_stopping != 'n':
        early_stopping = raw_input('    Please enter y (yes) or n (no): ')

    early_stopping = (early_stopping == 'y')

    monitor = None
    patience = None
    if early_stopping:
        monitor = raw_input('   What do you want to monitor (accuracy/loss)? ')
        while monitor != 'accuracy' and monitor != 'loss':
            monitor = raw_input('   Please enter "accuracy" or "loss": ')
        patience = raw_input('   Enter patience (leave blank for 2): ')
        if patience == '':
            patience = 2
        else:
            patience = int(patience)

    (train_acc, train_loss, val_acc, val_loss) = net.train(
        model_path,
        training_dataset_path,
        validation_dataset_path,
        early_stopping,
        monitor,
        patience)

    print("TRAINING FINISHED")

    if final_test:
        (test_dataset_path, output_matrix) = get_test_parameters()

        net.test(test_dataset_path, output_matrix, True, train_acc,
                 train_loss, val_acc, val_loss)


def get_test_parameters():
    '''Method to ask the user for the parameters of an incoming test of the
    network.

    Returns:
        test_dataset_path (str): entered path of the testing HDF5 dataset.
        output_matrix (str): desired name for the output matrix containing
        the results.'''
    test_dataset_path = raw_input(('Enter the path to the testing dataset ' +
                                   '.h5 file (leave blank to use standard ' +
                                   'MNIST): '))
    while test_dataset_path != '' and not os.path.isfile(test_dataset_path):
        test_dataset_path = raw_input(('    Please enter a correct path to ' +
                                       ' the .h5 file (leave blank to use ' +
                                       'standard MNIST): '))

    output_matrix = raw_input(('Enter the desired name for the output ' +
                               'matrix .mat file (leave blank to "results"): '
                               ))
    if output_matrix == '':
        output_matrix = 'results'

    output_matrix = (output_matrix + '.mat')

    return test_dataset_path, output_matrix


'''
NETWORK MANAGER: if this file is executed.
'''

if __name__ == '__main__':
    action = None
    while action != 'train' and action != 'test' and action != 'both':
        action = raw_input('\nWhat do you want to do (train/test/both)? ')

    if action == 'train':
        do_train()

    elif action == 'test':
        net = Network()
        model_path = raw_input(('Enter the path containing the model to ' +
                                'evaluate (leave blank for "my-model"): '))
        while model_path != '' and not os.path.exists(model_path):
            model_path = raw_input(('    Please enter a valid path (leave ' +
                                    'blank for "my-model")'))
        if model_path == '':
            model_path = 'my-model'

        net.load(model_path)

        (test_dataset_path, output_matrix) = get_test_parameters()
        net.test(test_dataset_path, output_matrix, False)

    else:
        do_train(final_test=True)
