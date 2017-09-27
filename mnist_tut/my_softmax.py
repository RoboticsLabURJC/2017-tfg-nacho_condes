from tensorflow.examples.tutorials.mnist import input_data # directly imports mnist DB

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True) # imports the dataset, and the labels as "one_hot" ([0,0,1,0,0,0...]) instead of nums.

import tensorflow as tf

x = tf.placeholder(tf.float32, [None, 784]) # placeholder for one image
# Parameters
W = tf.Variable(tf.zeros([784,10])) # tensor with weights for each pixel, evaluated on 10 classes (0-9)
b = tf.Variable(tf.zeros([10])) # vector for biases of each class


# MODEL
# y = tf.nn.softmax(tf.matmul(x, W) + b) # flipped multiplication to deal with x being a 2d tensor with inputs
y = tf.matmul(x,W) + b
# TRAINING
y_ = tf.placeholder(tf.float32, [None, 10]) # ground truth distribution (we will input the labels)
'''
we could now use the theoretical cross entropy expression:
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_*tf.log(y), reduction_indices=[1])) # reduce_sum = sum; reduce_mean = mean
numerically unstable: we will use this one instead
'''
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))

# optimization algorithm
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy) # very simple one

# now we launch the model
sess = tf.InteractiveSession()
# we have to initialize the variables!
tf.global_variables_initializer().run()

# training!!! we will do 1000 steps

for _ in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)    # we get a new batch of 100 random samples (with their corresponding labels) on each step
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys}) # we run the optimizer feeding it with the new batch
# EVALUATING
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1)) # argmax gives us the index of the most probable label in y, and the correct label in y_ (one_hot)
# the prediction is correct when they are equal. We will get an array of booleans (correct/incorrect)

# we will cast that array to float (0.0/1.0), and calculate the mean -> accuracy!!
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
# let's print it
print sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels})



