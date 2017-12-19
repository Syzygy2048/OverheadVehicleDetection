#based on https://www.tensorflow.org/get_started/mnist/beginners

#ignore compilation warnings concerning speed. carefull, might hide important warnings as well
import os
#os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('data/mnist', one_hot=True)

sess = tf.InteractiveSession()

input = tf.placeholder(tf.float32, shape=[None, 784], name="input")
groundTruth = tf.placeholder(tf.float32, shape=[None, 10], name="groundTruth")

weight = tf.Variable(tf.zeros([784,10]), name="weight")
bias = tf.Variable(tf.zeros([10]), name="bias")

sess.run(tf.global_variables_initializer())

output = tf.matmul(input, weight) + bias

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=groundTruth, logits=output))

trainStep = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

for step in range(1000):
    batch = mnist.train.next_batch(100)
    trainStep.run(feed_dict={input: batch[0], groundTruth: batch[1]})
    correct = tf.equal(tf.arg_max(output, 1), tf.arg_max(groundTruth, 1))

    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

    print(accuracy.eval(feed_dict={input: mnist.test.images, groundTruth: mnist.test.labels}))

