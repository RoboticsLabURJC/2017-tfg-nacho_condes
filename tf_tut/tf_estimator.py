#! /usr/bin/python2

'''
(2) Now, we will use tf.estimator frontend for modeling at high-level
'''
import tensorflow as tf
import numpy as np

feature_columns = [tf.feature_column.numeric_column("x", shape=[1])]

estimator = tf.estimator.LinearRegressor(feature_columns=feature_columns)

# training set
x_train = np.array([1., 2., 3., 4.])
y_train = np.array([0., -1., -2., -3.])
# evaluation set
x_eval = np.array([2., 5., 8., 1.])
y_eval = np.array([-1.01, -4.1, -7, 0.])

input_fn = tf.estimator.inputs.numpy_input_fn(
	{"x": x_train}, y_train, batch_size=4, num_epochs=None, shuffle=True)
train_input_fn = tf.estimator.inputs.numpy_input_fn(
	{"x": x_train}, y_train, batch_size=4, num_epochs=1000, shuffle=False)
eval_input_fn = tf.estimator.inputs.numpy_input_fn(
	{"x": x_eval}, y_eval, batch_size=4, num_epochs=1000, shuffle=False)

# now we invoke 1000 training steps by invoking the method and passing the
# training data set.
estimator.train(input_fn=input_fn, steps=1000)

# evaluation
train_metrics = estimator.evaluate(input_fn=train_input_fn)
eval_metrics = estimator.evaluate(input_fn=eval_input_fn)
print("train metrics: %r"% train_metrics)
print("eval metrics: %r"% eval_metrics)


'''
output:
train metrics: {'average_loss': 1.1186788e-07, 'global_step': 1000, 'loss': 4.4747151e-07}
eval metrics: {'average_loss': 0.0025502923, 'global_step': 1000, 'loss': 0.010201169}

the eval loss is higher, but close to zero: things are going properly
'''
