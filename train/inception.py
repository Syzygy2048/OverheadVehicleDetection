from collections import deque

import tensorflow as tf
from tensorflow.contrib.slim.nets import inception as nn_architecture
from tensorflow.contrib import slim

import time

import numpy as np
from skimage import io

import copy

CHECKPOINT_PATH = "C:/Users/Work/Documents/OverheadVehicleDetection/checkpoints\inception_v3.ckpt"  # https://github.com/tensorflow/models/tree/master/research/slim#pre-trained-models

images = tf.placeholder(tf.float32, shape=[None, 299, 299, 3], name='images')
labels = tf.placeholder(tf.float32, shape=[None, 2], name='labels')

with slim.arg_scope(
        nn_architecture.inception_v3_arg_scope()):  # https://kwotsin.github.io/tech/2017/02/11/transfer-learning.html
    logits, endpoints = nn_architecture.inception_v3(images,  # input
                                                     num_classes=2,
                                                     is_training=True)

retrain = ['InceptionV3/Logits', 'InceptionV3/AuxLogits', 'InceptionV3/Mixed_7c']  # could also remove some of the input channels (probably Conv2d_1a_3x3) to get rid of 3 color channel inputs and accept monochrome
variables_to_restore = slim.get_variables_to_restore(
    exclude=retrain)  # this checks the current graph, so no custom nodes can be defined at this point, only those from create_network()

sess = tf.Session()
saver = tf.train.Saver(variables_to_restore)
saver.restore(sess, CHECKPOINT_PATH)
loss = tf.losses.softmax_cross_entropy(onehot_labels=labels, logits=logits)
total_loss = tf.losses.get_total_loss()

global_step = tf.train.get_or_create_global_step()

optimizer = tf.train.AdamOptimizer(1e-4)
variables_to_train = tf.trainable_variables('InceptionV3/Logits') + tf.trainable_variables('InceptionV3/AuxLogits') + tf.trainable_variables('InceptionV3/Mixed_7c')

train_step = slim.learning.create_train_op(total_loss, optimizer, variables_to_train=variables_to_train)  # better than Optimizer.minimize because it prevents problems like vanishing gradient

correct = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))  # converts one hot output vector to percentage

sess.run(tf.global_variables_initializer())

# tensorboard logging
summary_writer = tf.summary.FileWriter("C:/Users/Work/Documents/OverheadVehicleDetection/log/tensorboard", sess.graph)

tf.summary.scalar('total loss', total_loss)
tf.summary.scalar('loss', loss)
tf.summary.scalar('accuracy', accuracy)

summary_op = tf.summary.merge_all()

anno_path_car = "C:/Users/Work/Documents/OverheadVehicleDetection/data/cowc_preprocessed/Toronto/3553/car/px40/%d.png"
anno_path_neg = "C:/Users/Work/Documents/OverheadVehicleDetection/data/cowc_preprocessed/Toronto/3553/neg/px40/%d.png"

batchsize = 200

anno_amount_car = 2388 #this is valid for toronto 3553 40px
queue_car_template = deque()
for i in range(anno_amount_car):
    queue_car_template.append(i)
queue_car = copy.copy(queue_car_template)



anno_amount_neg = 7449 #this is valid for toronto 3553 40px
queue_neg_template = deque()
for i in range(anno_amount_neg):
    queue_neg_template.append(i)
queue_neg = copy.copy(queue_neg_template)

epoch_size =  anno_amount_car + anno_amount_neg
epoch_car = 0
epoch_neg = 0

for i in range(0, epoch_size * 50000, batchsize):
# while True:
    time1 = time.time()

    imgs = np.empty(shape=[0, 299, 299, 3])
    lbls = np.empty(shape=[0, 2])

    for j in range(int(batchsize/2)):
        while True:
            try:
                if len(queue_car) < 1:
                    queue_car = copy.copy(queue_car_template)
                    epoch_car = epoch_car + 1
                img = io.imread(anno_path_car % queue_car.pop())
                imgs = np.vstack([imgs, [img]])
                lbls = np.vstack([lbls, [1, 0]])
                break
            except FileNotFoundError:
                continue
        while True:
            try:
                if len(queue_neg) < 1:
                    queue_neg = copy.copy(queue_neg_template)
                    epoch_neg = epoch_neg + 1
                img = io.imread(anno_path_neg % queue_neg.pop())
                imgs = np.vstack([imgs, [img]])
                lbls = np.vstack([lbls, [0, 1]])
                break
            except FileNotFoundError:
                continue

    _, summary = sess.run([train_step, summary_op], feed_dict={images: imgs, labels: lbls})
    summary_writer.add_summary(summary)
    with sess.as_default():
        train_accuracy = accuracy.eval(feed_dict={images: imgs, labels: lbls})
        # logis = logits.eval(feed_dict={images: imgs, labels: lbls})
        # print(logis)
        print('%d.%d.%d - %d | accuracy %g - %d/%d' % (epoch_car, epoch_neg, i/batchsize, time.time() - time1, train_accuracy, batchsize * train_accuracy, batchsize))



# XPS-15, batchsize 200, dataset 3553
# Full Training: dT - 350 sec, accuracy 0.88,
# Retrain Logits and AuxLogits: dt - 180 sec, accuracy 0.5
#
#