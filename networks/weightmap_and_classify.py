''' This script loads MNIST Dataset, and trains a monolayer network with it, 
After that (you can play with the UPPERCASE parameters, which configure the training process),
we are able to print the learned weights for each pixel on each class (0-9) on a heat map.
At the end of the script, we can enter an image (extracted from the dataset) to the network, so it
returns the label which it predicts for that particular image. This is a beginning for my project!

Thanks to https://www.oreilly.com/learning/not-another-mnist-tutorial-with-tensorflow
'''

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
import matplotlib.pyplot as plt
import numpy as np
import random as ran

def TRAIN_SIZE(num):
    print('Total Training Images in Dataset = ' +
            str(mnist.train.images.shape))
    print('----------------------------------------------')
    x_train = mnist.train.images[:num,:]
    print('x_train Examples Loaded = ' + str(x_train.shape))
    y_train = mnist.train.labels[:num,:]
    print('y_train Examples Loaded = ' + str(y_train.shape))
    print('')
    return x_train, y_train

def TEST_SIZE(num):
    print('Total Test Examples in Dataset = ' + str(mnist.test.images.shape))
    print('---------------------------------------------')
    x_test = mnist.test.images[:num,:]
    print('x_test Examples Loaded = ' + str(x_test.shape))
    y_test = mnist.test.labels[:num,:]
    print('y_test Examples Loaded = ' + str(y_test.shape))
    return x_test, y_test

def display_digit(num):
    print(y_train[num])
    label = y_train[num].argmax(axis=0)
    image = x_train[num].reshape(28,28)
    plt.title('Example: %d Label: %d' % (num,label))
    plt.imshow(image, cmap=plt.get_cmap('gray_r'))
    plt.show()

def display_mult_flat(start, stop):
    images = x_train[start].reshape([1,784])
    for i in range(start + 1, stop):
        images = np.concatenate((images, x_train[i].reshape([1, 784])))
    plt.imshow(images, cmap=plt.get_cmap('gray_r'))
    plt.show()

x_train, y_train = TRAIN_SIZE(50000)
display_digit(ran.randint(0,x_train.shape[0]))
'''
x_train, y_train = TRAIN_SIZE(50000)
display_digit(ran.randint(0,x_train.shape[0]))

Display a random digit -> display_digit(ran.randint(0, x_train.shape[0]))
Display 400 digits in flattened way ->  display_mult_flat(0,400)
'''
import tensorflow as tf
sess = tf.Session()

x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])

W = tf.Variable(tf.zeros([784,10])) # Like a 'cheat sheet' for each pixel/class
b = tf.Variable(tf.zeros([1, 10]))

# evaluation function
y = tf.nn.softmax(tf.matmul(x,W) + b)

# loss function
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

x_train, y_train = TRAIN_SIZE(10000)
x_test, y_test = TEST_SIZE(1000)

LEARNING_RATE = 0.01
TRAIN_STEPS = 50000

init = tf.global_variables_initializer()
sess.run(init)

training = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


for i in range(TRAIN_STEPS + 1):
    sess.run(training, feed_dict={x: x_train, y_: y_train})
    if i % 100 == 0:
        print('Training Step: ' + str(i) + ' Accuracy = ' + str(sess.run(accuracy, feed_dict={x: x_test, y_: y_test})) + ' Loss = ' +
                str(sess.run(cross_entropy, {x: x_train, y_: y_train})))


# TRAINING DONE. "OPTIMAL" VALUES FOUND
for i in range(10):
    plt.subplot(2, 5, i+1)
    weight = sess.run(W)[:,i]
    plt.title(i)
    plt.imshow(weight.reshape([28,28]), cmap = plt.get_cmap('seismic'))
    frame1 = plt.gca()
    frame1.axes.get_xaxis().set_visible(False)
    frame1.axes.get_yaxis().set_visible(False)

plt.show()



x_train, y_train = TRAIN_SIZE(1)
display_digit(0)


answer = sess.run(y, feed_dict={x: x_train})
#print(sess.run(tf.argmax(answer,1)))
print answer.argmax() # it prints "7"!! (the predicted label, which is correct)

