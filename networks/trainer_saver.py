# Loading data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot= True)
# stores everything as np arrays, and provides a funct. to iterate
import tensorflow as tf

# placeholders (valores de entrada)
x = tf.placeholder(tf.float32, shape=[None, 784]) # una imagen de 28x28 flattened
y_ = tf.placeholder(tf.float32, shape=[None,10]) # etiquetas de salida de la red (one_hot)


# initializing it with noise
def weight_variable(shape):
	initial = tf.truncated_normal(shape, stddev=0.1)
	return tf.Variable(initial)

def bias_variable(shape):
	initial = tf.constant(0.1, shape=shape)
	return tf.Variable(initial)

def conv2d(x, W):
	return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
	return tf.nn.max_pool(x, ksize=[1, 2, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

# Variables
W_conv1 = weight_variable([5, 5, 1, 32]) # weights for each pixel in each class
b_conv1 = bias_variable([32]) # biasing for each class
'''
We resize the images batch to a 4D tensor -> (infer (n. of images), width, height, channels)
This is because the conv2d function requires the input exactly in this way
'''
x_image = tf.reshape(x, [-1, 28, 28, 1])

# outputs, operations
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1) # W acts as a filter
h_pool1 = max_pool_2x2(h_conv1)

W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# Fully Connected Layer
W_fc1 = weight_variable([7*7*64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)


W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2


LEARNING_RATE = 1e-4
TRAIN_STEPS = 2000

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
train_step = tf.train.AdamOptimizer(LEARNING_RATE).minimize(cross_entropy)
accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1)), tf.float32))

saver = tf.train.Saver()

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	for step in range(TRAIN_STEPS):
	    batch = mnist.train.next_batch(10)
	    sess.run(train_step, feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
	    if step % 100 == 0:
	        print("step: %d, accuracy: %3f" % (step, sess.run(accuracy, feed_dict= {x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0})))
	        path = saver.save(sess, './my-model/model', global_step=step)

	print("training completed in:%s" % (path))
