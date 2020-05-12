# -*- coding: utf-8 -*-
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

# let's generate a model dataset
N = 1000
D = 3
x = np.random.random((N, D))
w = np.random.random((D, 1))
y = x @ w + np.random.randn(N, 1) * 0.20

# placeholders for input data of linear reg
tf.reset_default_graph()
features = tf.placeholder(tf.float32, shape=(None, D)) # N=1000, D=3
target = tf.placeholder(tf.float32, shape=(None, 1))

# make predictions
weights = tf.get_variable('w', shape=(D, 1), dtype=tf.float32)
predictions = features @ weights

# define our loss
loss = tf.reduce_mean((target-predictions) ** 2)

# optimizer 
optimizer = tf.train.GradientDescentOptimizer(0.1)
step = optimizer.minimize(loss)

# gradient descent step
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
for i in range(300):
    _, curr_loss, curr_weights = sess.run([step, loss, weights],
                                          feed_dict={features:x, target:y}) # feeding placeholders
    if i%50==0:
        print(curr_loss)
        
# close the session
tf.Session.close(sess)
