# -*- coding: utf-8 -*-

#%%
import tensorflow.compat.v1 as tf
import sys
sys.path.append("../")
from keras_utils import reset_tf_session
#%%
tf.disable_v2_behavior()
sess = reset_tf_session()
print('we are using Tf', tf.__version__)
#%%
## Wariming up
import numpy as np
def sum_python(N):
    return np.sum(np.arange(N) ** 2)
#%%
sum_python(10**5)
#%%
# Tensorflow teaser

# Initialize the parameter
N = tf.placeholder('int64', name='input_to_fun')

# a recipe how to produce result
result = tf.reduce_sum(tf.range(N) ** 2)
result
#%%
result.eval({N: 10**5})
# logger for the tensorboard
writer = tf.summary.FileWriter('Tensorboard_logs', graph=sess.graph)
#%%
with tf.name_scope('Placeholder_examples'):
    # default placeholder that can be arobitrary float32
    # scalar vector, matirx etc
    arbitrary_input = tf.placeholder('float32')
    
    #input_vector of arbitrary length
    input_vector = tf.placeholder('float32', shape=(None, ))
    
    # input vector that must have 10 elements and integer type
    fixed_vector = tf.placeholder('float32', shape=(10, ))
    
    # Matrix of arbitrary n_rows and 15 columns 
    input_matrix = tf.placeholder('float32', shape=(None, 15))
    
    # None can be used for any non-specific shape
    input1 = tf.placeholder('float64', shape=(None, 100, None))
    input2 = tf.placeholder('int32', shape=(None, None, 3, 224, 224))
    
    # elementwise multiplication
    double_the_vector = input_vector ** 2
    
    # elementwise multiplication 
    elementwise_cosine = tf.cos(input_vector)
    
    # diff bet squarred vetor and vector itself plus one
    vector_sq = input_vector**2 - input_vector + 1
    
#%%
my_vector = tf.placeholder('float32', shape=(None, ), name='VECTOR_1')
my_vector2 = tf.placeholder('float32', shape=(None, ))
my_transformation = my_vector * my_vector2/(tf.sin(my_vector) + 1)
#%%
print(my_transformation) #prints the graph instance

#%%
dummyData = np.arange(5).astype('float32')
print(dummyData)
my_transformation.eval({my_vector:dummyData, my_vector2:dummyData[::-1]})
#%
writer.add_graph(my_transformation.graph)
writer.flush()
################################################################################################
# Tensorlfow is based on computaion graphs
# A graph consists of placeholders and transformations
################################################################################################

#%%
# Implement LOss function : MSE
with tf.name_scope('MSE'):
    y_true = tf.placeholder('int32', shape=(None, ), name='y_true')
    y_pred = tf.placeholder('int32', shape=(None, ), name='y_predicted')
    mse = tf.reduce_sum(tf.square(y_true - y_pred))
    
#%%
def compute_mse(vector1, vector2):
    return mse.eval({y_true:vector1, y_pred:vector2})
writer.add_graph(mse.graph)
writer.flush()

#%%
# Rigorous local testing
import sklearn.metrics
for n in [1, 5, 10, 10**3]:
    elems = [np.arange(n), np.arange(n, 0, -1), np.zeros(n), np.ones(n), np.random.random(n), np.random.randint(100, size=n)]
    for el in elems:
        for el_2 in elems:
            true_mse = np.array(sklearn.metrics.mean_squared_error(el, el_2))
            my_mse = compute_mse(el, el_2)
            
            if not np.allclose(true_mse, my_mse):
                print('mse(%s, %s)' %(el, el_2))
                print("should be: %f, but our function returned %f" %(true_mse, my_mse))
                raise ValueError('Wrong result')

#%% 
#deal with it later, finish up the notebook first
# Variables
# Creating a sharef variable
shared_vector1 = tf.Variable(initial_value=np.ones(5), name='example_variable')
# initialize variable with initial values
sess.run(tf.global_variables_initializer())

# evaluating the shared variable
print('Initial Value:', sess.run(shared_vector1))
#%%
# setting a new value
sess.run(shared_vector1.assign(np.arange(5)))

# get the new value
print("New Value", sess.run(shared_vector1))

#%%
## tf.gradients - why graphs matter 
my_scalar = tf.placeholder('float32')
scalar_squarred = my_scalar ** 2
# derivative of scalar-squared by my_scalar
derivative = tf.gradients(scalar_squarred, [my_scalar, ])
derivative

#%%
import matplotlib.pyplot as plt

x = np.linspace(-3, 3)
x_squarred, x_squared_der = sess.run([scalar_squarred, derivative[0]],
                                     {my_scalar:x})
plt.plot(x, x_squarred, label='$x^2$')
plt.plot(x, x_squared_der, label=r"$\frac{d(x^2)}{dx}$")
plt.legend()

#%%
my_vector = tf.placeholder('float32', [None])
# Compute the gradient of the next weird function over my_scalar and my_vector
# Warning! Trying to understand the meaning of that function may result in permanent brain damage
weird_psychotic_function = tf.reduce_mean(
    (my_vector+my_scalar)**(1+tf.nn.moments(my_vector,[0])[1]) + 
    1./ tf.atan(my_scalar))/(my_scalar**2 + 1) + 0.01*tf.sin(
    2*my_scalar**1.5)*(tf.reduce_sum(my_vector)* my_scalar**2
                      )*tf.exp((my_scalar-4)**2)/(
    1+tf.exp((my_scalar-4)**2))*(1.-(tf.exp(-(my_scalar-4)**2)
                                    )/(1+tf.exp(-(my_scalar-4)**2)))**2

der_by_scalar = tf.gradients(weird_psychotic_function, my_scalar)
der_by_vector = tf.gradients(weird_psychotic_function, my_vector)

#%%
# all I can do is , plot and see and leave
scalar_space = np.linspace(1, 7, 100)
y = [sess.run(weird_psychotic_function, {my_scalar:x, my_vector:[1, 2, 3]}) for x in scalar_space]

plt.plot(scalar_space, y, label='function')
y_der_by_scalar = [sess.run(der_by_scalar,
                            {my_scalar:x, my_vector:[1, 2, 3]}) for x in scalar_space]

plt.plot(scalar_space, y_der_by_scalar, label='derivative')
plt.grid()
plt.legend()

#%%
# Optimizers
y_pred = tf.Variable(np.zeros(2, dtype='float32'))
y_true = tf.range(1, 3, dtype='float32')

loss = tf.reduce_mean((y_pred - y_true + 0.5 * tf.random_normal([2]))**2)

step = tf.train.MomentumOptimizer(0.03, 0.5).minimize(loss, var_list=y_pred)
    
#%%
from matplotlib import animation, rc
import matplotlib_utils
#from Ipython.display import HTML, display_html

# nice figure settings
fig, ax = plt.subplots()
y_true_value = sess.run(y_true)
level_x = np.arange(0, 2, 0.02)
level_y = np.arange(0, 3, 0.02)
X, Y = np.meshgrid(level_x, level_y)
Z = (X - y_true_value[0])**2 + (Y - y_true_value[1]) ** 2
ax.set_xlim(-0.02, 2)
ax.set_ylim(-0.02, 3)
sess.run(tf.global_variables_initializer())
ax.scatter(*sess.run(y_true), c='red')
contour = ax.contour(X, Y, Z, 10)
ax.clabel(contour, inline=1, fontsize=10)
line, = ax.plot([], [], lw=2)

# start animation with empty trajectory
def init():
    line.set_data([], [])
    return (line, )
trajectory = [sess.run(y_pred)]

# one animation step
def animate(i):
    sess.run(step)
    trajectory.append(sess.run(y_pred))
    line.set_data(*zip(*trajectory))
    return (line, )
anim = animation.FuncAnimation(fig, animate, init_func=init, frames=100, interval=20, blit=True)
    