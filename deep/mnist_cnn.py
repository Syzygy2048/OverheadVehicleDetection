#based on https://www.tensorflow.org/get_started/mnist/pros
import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('data/mnist', one_hot=True)

input = tf.placeholder(tf.float32, shape=[None, 784], name="input")
groundTruth = tf.placeholder(tf.float32, shape=[None, 10], name="groundTruth")



#helper methods to avoid identical weights and biases.
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')



#first layer
W_conv1 = weight_variable([5, 5, 1, 32]) # 5,5, is patch size, 1 is number of input channels, 32 is the number of output channels
b_conv1 = bias_variable([32]) #bias, one for each output channel


x_image = tf.reshape(input, [-1, 28, 28, 1]) #make a 4d vector out of the 3d data (reshaped, x dim, y dim, color channels)

#apply first layer
h_image = tf.nn.relu(conv2d(x_image, W_conv1,) + b_conv1) # w * x + b, ReLU sets all negative values to 0 and keeps the positive ones, values can be negative because weight and bias can be negative
h_pool1 = max_pool_2x2(h_image) #reduces resulution by half -> 14


#second layer
W_conv2 = weight_variable([5,5,32,64]) #same patch size, all output layers from the previous layer are input layers, double the output layers
b_conv2 = bias_variable([64])

#apply second layer
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2) #reduces resolution by half -> 7


#third layer, fully connected
W_fc1 = weight_variable([7*7*64, 1024])
b_fc1 = bias_variable([1024])

#apply thrid layer
h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)


#dropout random weights and biases to prevent overfitting
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

#output layer
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

output = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

###############
## chuu chuu ##
###############

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=groundTruth, logits=output))
train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)
correct = tf.equal(tf.argmax(output, 1), tf.argmax(groundTruth, 1))
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32)) #converts one hot output vector to percentage


with tf.Session() as sess:
    print("start of session")
    sess.run(tf.global_variables_initializer())
    for i in range(20000):
        #print("train step %d" % (i))
        batch = mnist.train.next_batch(50)
        if (i % 100 == 0):
            train_accuracy = accuracy.eval(feed_dict={input: batch[0], groundTruth: batch[1], keep_prob: 1.0})
            print('step %d, training accuracy %g' % (i, train_accuracy))
        train_step.run(feed_dict={input: batch[0], groundTruth: batch[1], keep_prob: 1.0})
    print('test accuracy %g' % accuracy.eval(feed_dict={input: mnist.test.images, groundTruth: mnist.test.labels, keep_prob: 1.0}))