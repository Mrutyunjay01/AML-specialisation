# -*- coding: utf-8 -*-
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


# a variable has an initial value. You need to initialize it in the graph execution env
# define the graph
tf.reset_default_graph()
a = tf.constant(np.ones((2, 2)), dtype=tf.float32)
b = tf.Variable(tf.ones((2, 2)))
c = a @ b

# Create a session
sess = tf.InteractiveSession()
'''
# furious runnig attempt
sess = tf.InteractiveSession()
sess.run(c)
#############
So, yeah! It throws 'Attempting to use uninitialized value variable'
'''
# initialise the variables inside the graph
sess.run(tf.global_variables_initializer())
print(sess.run(c))

# close the session
tf.Session.close(sess)