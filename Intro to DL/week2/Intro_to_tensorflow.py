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