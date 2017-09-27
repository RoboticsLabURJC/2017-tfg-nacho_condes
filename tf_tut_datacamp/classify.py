import preproc
import numpy as np
import tensorflow as tf
import random
import matplotlb.pyplot as plt

images28, labels = preproc.preproc()


# Initialize placeholders
x = tf.placeholder(dtype = tf.float32, shape = [None, 28, 28])
y = tf.placeholder(dtype = tf.int32, shape = [None])

# Flatten the input data ("MATLAB reshape": [None, 28, 28] => [None, 784])
images_flat = tf.contrib.layers.flatten(x)

# Fully connected layer (logit indicates that this tensor will be mapped to probs. (for softmax))
logits = tf.contrib.layers.fully_connected(images_flat, 62, tf.nn.relu)

# Define a loss function
loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels = y, logits = logits))

# Define an optimizer
train_op = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

# Convert logits to label indexes

correct_pred = tf.argmax(logits, 1)

# Define an accuracy metric
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
'''
print("images_flat: ", images_flat)
print("logits: ", logits)
print("loss: ", loss)
print("predicted_labels: ", correct_pred)
'''
# NEURAL NETWORK FEEDING
#
#
tf.set_random_seed(1234)
sess = tf.Session()

sess.run(tf.global_variables_initializer())


















