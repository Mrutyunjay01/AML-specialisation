# -*- coding: utf-8 -*-
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

# create the placeholders
tf.reset_default_graph()
a = tf.placeholder(np.float32, (2, 2))
b = tf.Variable(tf.ones((2, 2)))
c = a @ b


# create a session
sess = tf.InteractiveSession()
'''
# Runnig the wrong method
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
print(sess.run(c))
###########
So, as we saw:
    Invalid argument: You must feed a value for placeholder tensor 'Placeholder' with dtype float and shape [2,2]
Hence, use feed_dict method to feed a placeholder
'''
sess.run(tf.global_variables_initializer())
print(sess.run(c, feed_dict={a : np.ones((2, 2))}))
# close the session
tf.Session.close(sess)
