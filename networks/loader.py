from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot= True)

import tensorflow as tf

# we have to re-declare variables before restoring them
x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None,10])

W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))

y = tf.matmul(x,W) + b
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
								labels=y_, logits=y))

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

saver = tf.train.Saver()

with tf.Session() as sess:
	saver.restore(sess, './my-model/model-900')
	print("Restored!")
	test_acc = sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels})
	print("Accuracy")
	print(test_acc)
