# -*- coding: utf-8 -*-
# disable v2 behaviour as place_holder() and a few more functions have been removed from v2
import tensorflow.compat.v1 as tf
import numpy as np

tf.disable_v2_behavior()

# creating a placeholder of given data type and shape of tensors
x = tf.placeholder(tf.float32, (None, 10))

# creating a variable
p = tf.get_variable('p', shape=(10, 20), dtype=tf.float32)
p_1 = tf.Variable(tf.random_uniform((10, 20)), name='p_1')

# creating a constant
c = tf.constant(np.ones((4, 4)))

#Operation 
z = x @ p # Multiplicaton
print(z)


# to get the operation
tf.get_default_graph().get_operations()
# to get the output
tf.get_default_graph().get_operations()[0].outputs
#tf.reset_default_graph()