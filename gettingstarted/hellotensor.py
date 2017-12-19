#this file is used to figure out the most basic TF and python syntax and behavior.

#ignore compilation warnings concerning speed. carefull, might hide important warnings as well
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import tensorflow as tf
import math

node1 = tf.constant(3.0)
node2 = tf.constant(4.0)

sess = tf.Session()
print(sess.run([node1, node2]))

node3 = node1 * node2

print("node3:", node3)
print("sess.run(node3):", sess.run(node3))


a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)

c = tf.sqrt(tf.pow(a, 2.0) + tf.pow(b, 2.0))

print("tf placeholder pyth  -", sess.run(c, {a:3, b:4}))

print("tf placeholder pyth2 -", sess.run(c, {a:[3, 9], b:[4, 12]}))

d = tf.pow(c, 4.0)

print("tf flow ", sess.run(d, {a:3, b:4}))

#########################
## create linear model ##
#########################

w = tf.Variable([0.1], "weight")
b = tf.Variable([-0.3], "bias")
x = tf.placeholder(dtype=tf.float32, name="input")

linear_model = w*x+b


#inviitalize variables w and b, because defining them is not enough
init = tf.global_variables_initializer()
sess.run(init)

#run untrained model
print(sess.run(linear_model, {x:[1,2,3,4]}))

y = tf.placeholder(dtype=tf.float32, name="groundTruth")

#create tf node for error and loss functions
squarederror = tf.square(linear_model - y)
loss = tf.reduce_sum(squarederror)

#calculate loss with initial values
print("loss: ", sess.run(loss, {x:[1,2,3,4], y:[0,-1,-2,-3]}))

#reassign variable to show how it's done
wreassigned = tf.assign(w, [0.3])
sess.run([wreassigned]) #need to run the reassignment

trainX = [1,2,3,4]
trainY = [0,-1,-2,-3]
print("loss: ", sess.run(loss, {x:trainX, y:trainY}))
#note that this reassignment permanently adds a node to the tf graph

########################
## train linear model ##
########################

learnrate = 0.01
optimizer = tf.train.GradientDescentOptimizer(learnrate)
train = optimizer.minimize(loss)

sess.run(init)
#running  init resets to defaults, so the reassignment of w is undone
print("loss: ", sess.run(loss, {x:trainX, y:trainY}))

for i in range(1000):
    sess.run(train, {x:trainX, y:trainY})
    #w, b, l = sess.run([weight, bias, loss], {y: trainY, x: trainX})
    #print("weight: %s, bias %s, loss %s" % (w, b, l))
    #print("weight: %s, bias %s, loss %s" % (w.eval(session=sess), b.eval(session=sess), loss.eval(session=sess)))


w, b, l = sess.run([w, b, loss], {y: trainY, x: trainX})
print("weight: %s, bias %s, loss %s" % (w, b, l))

