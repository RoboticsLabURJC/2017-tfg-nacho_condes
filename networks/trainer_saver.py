# Loading data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot= True)
# stores everything as np arrays, and provides a funct. to iterate
import tensorflow as tf
sess = tf.Session() # interactive session (flexibility)

# placeholders (valores de entrada)
x = tf.placeholder(tf.float32, shape=[None, 784]) # una imagen de 28x28 flattened
y_ = tf.placeholder(tf.float32, shape=[None,10]) # etiquetas de salida de la red (one_hot)

# Variables
W = tf.Variable(tf.zeros([784,10])) # weights for each pixel in each class
b = tf.Variable(tf.zeros([10])) # biasing for each class

sess.run(tf.global_variables_initializer()) # initialization

# regression model
y = tf.nn.softmax(tf.matmul(x,W) + b)

# loss function
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

LEARNING_RATE = 0.01
TRAIN_STEPS = 50000

train_step = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(cross_entropy)

accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1)), tf.float32))

saver = tf.train.Saver()

for step in range(10000):
    batch = mnist.train.next_batch(10000)
    sess.run(train_step, feed_dict={x: batch[0], y_: batch[1]})
    if step % 100 == 0:
        print("step: %d, accuracy: %3f" % (step, sess.run(accuracy, feed_dict= {x: mnist.test.images, y_: mnist.test.labels})))
        path = saver.save(sess, './my-model/model', global_step=step)

print("training completed in:%s" % (path))
# EVALUATION
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1)) # list of booleans.
# we have to cast the boolean vector to 1-0 vector

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

print accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels})
# 92 % accuracy (bad result)
