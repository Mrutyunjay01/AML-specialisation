# -*- coding: utf-8 -*-

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

# tf.session encapsulates the env in which
# tf.operation objects are execud and
# tf.tensor objects are evaluated

# craete a session
sess = tf.InteractiveSession()

# define a graph
a = tf.constant(5.0)
b = tf.constant(6.0)
c = a * b

# running a graph
print(c)
print(sess.run(c))

# always close the session
tf.Session.close(sess)