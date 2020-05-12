import numpy as np
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()

# create the graph
tf.reset_default_graph()
x = tf.get_variable('x', shape=(), dtype=tf.float32)
f = x ** 2
# logging with tf.Print
f = tf.Print(f, [x, f], "x, f:")
# say we want to minimize the function f
optimizer = tf.train.GradientDescentOptimizer(0.1)
step = optimizer.minimize(f)
# as all the variables are trainable by defualt with 'trainable'm positional 
# argument in variable scope, we dont neeed to specify again.
# we can get all the trainable variables as follows: tf.trainable_variables()

# Making gd steps
# create a session and initialize the variables
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

# say we want to take 10 gd steps
for i in range(10):
    #_, curr_x, curr_f = sess.run([step, x, f]) #1st element prints None, Hence ignored
    #print(curr_x, curr_f)
    print(sess.run([step, f]))
#close the session , You know : safe practice
tf.Session.close(sess)