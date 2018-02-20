# based  on https://medium.com/initialized-capital/we-need-to-go-deeper-a-practical-guide-to-tensorflow-and-inception-50e66281804f
# and https://kwotsin.github.io/tech/2017/02/11/transfer-learning.html

import tensorflow as tf
import time

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('data/mnist', one_hot=True)

from tensorflow.contrib.slim.nets import inception as nn_architecture
from tensorflow.contrib import slim


import numpy as np
from scipy.ndimage.interpolation import zoom

NUM_CLASSES = 10
CHECKPOINT_PATH = "checkpoints/inception_v3.ckpt" #https://github.com/tensorflow/models/tree/master/research/slim#pre-trained-models
BATCH_SIZE = 25



MEAN = np.mean(mnist.train.images)
STD = np.std(mnist.train.images)
NUM_TRAIN = mnist.train.labels.shape[0]
NUM_TEST = mnist.test.labels.shape[0]

print("train data %d, test data %d" % (NUM_TRAIN, NUM_TEST))


# A convenience method for resizing the 784x1 monochrome images into
# the 299x299x3 RGB images that the Inception model accepts as input
RESIZE_FACTOR = (299 / 28)
def resize_images(images, mean=MEAN, std=STD):
    reshaped = (images - mean) / std    # why? - (standard normal distribution)this causes the mean to be 0 and the variance to be 1, resulting in better recognition results - technically not resizing, but w/e, the mean and std are calculatied from the whole trainings set, and are applied to everything put into the network. training and production. #### should be backed by a source? i was told this by someone in ##machinelearning on freenode
                                        # additional infos at: https://www.tensorflow.org/tutorials/image_recognition#usage_with_the_c_api
                                        # We also need to scale the pixel values from integers that are between 0 and 255 to the floating point values that the graph operates on. We control the scaling with the input_mean and input_std flags: we first subtract input_mean from each pixel value, then divide it by input_std.
                                        # These values probably look somewhat magical, but they are just defined by the original model author based on what he/she wanted to use as input images for training. If you have a graph that you've trained yourself, you'll just need to adjust the values to match whatever you used during your training process.

    reshaped = np.reshape(reshaped, [-1, 28, 28, 1])  # Reshape 784 to 28x28x1

    # Reshape to 299x299 images, then duplicate the single monochrome channel
    # across 3 RGB layers
    resized = zoom(reshaped, [1.0, RESIZE_FACTOR, RESIZE_FACTOR, 1.0])
    resized = np.repeat(resized, 3, 3)  # add color channels

    return resized

images = tf.placeholder(tf.float32, shape=[None, 299, 299, 3], name='images')
labels = tf.placeholder(tf.float32, shape=[None, 10], name='labels')




with slim.arg_scope(nn_architecture.inception_v3_arg_scope()):  # https://kwotsin.github.io/tech/2017/02/11/transfer-learning.html
    logits, endpoints = nn_architecture.inception_v3(images,  # input
                                                     num_classes=10,
                                                     is_training=True)

retrain = ['InceptionV3/Logits', 'InceptionV3/AuxLogits', 'InceptionV3/Mixed_7c', 'InceptionV3/Mixed_7b']  # could also remove some of the input channels (probably Conv2d_1a_3x3) to get rid of 3 color channel inputs and accept monochrome
variables_to_restore = slim.get_variables_to_restore(exclude=retrain)  # this checks the current graph, so no custom nodes can be defined at this point, only those from create_network()

sess = tf.Session()
saver = tf.train.Saver(variables_to_restore)
saver.restore(sess, CHECKPOINT_PATH)

loss = tf.losses.softmax_cross_entropy(onehot_labels=labels, logits=logits)
total_loss = tf.losses.get_total_loss()


global_step = tf.train.get_or_create_global_step()

optimizer = tf.train.AdamOptimizer(1e-4)
variables_to_train = tf.trainable_variables('InceptionV3/Logits') + tf.trainable_variables('InceptionV3/AuxLogits') + tf.trainable_variables('InceptionV3/Mixed_7c') + tf.trainable_variables('InceptionV3/Mixed_7b')
print('variables to train', variables_to_train)
train_step = slim.learning.create_train_op(total_loss, optimizer, variables_to_train=variables_to_train) #better than Optimizer.minimize because it prevents problems like vanishing gradient

correct = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32)) #converts one hot output vector to percentage

sess.run(tf.global_variables_initializer())

#tensorboard logging
tf.summary.FileWriter("log/tensorboard", sess.graph)

for i in range(5000):
    time1 = time.time()
    batch = mnist.train.next_batch(BATCH_SIZE)
    img = resize_images(batch[0])
    lbls = batch[1]
    sess.run([train_step], feed_dict={images: img, labels: lbls})
    with sess.as_default():
            train_accuracy = accuracy.eval(feed_dict={images: img, labels: lbls})
            print('step %d, training accuracy %g, time: %s' % (i, train_accuracy, time.time() - time1))