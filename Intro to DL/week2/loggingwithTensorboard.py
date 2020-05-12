# -*- coding: utf-8 -*-

import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()

# create the graph
tf.reset_default_graph()
x = tf.get_variable('x', shape=(), dtype=tf.float32)
f = x ** 2

# say we want to minimize the function f
optimizer = tf.train.GradientDescentOptimizer(0.1)
step = optimizer.minimize(f)

# add summeries
tf.summary.scalar('curr_x', x)
tf.summary.scalar('curr_f', f)
summaries = tf.summary.merge_all()

# logging the summeries
sess = tf.InteractiveSession()
summary_writer = tf.summary.FileWriter('logs/1', sess.graph) #run number is 1
sess.run(tf.global_variables_initializer())

# taking 10 gd steps
for i in range(10):
    _, curr_summaries = sess.run([step, summaries])
    summary_writer.add_summary(curr_summaries, i)
    summary_writer.flush()

# fucking close the session
tf.Session.close(sess)
