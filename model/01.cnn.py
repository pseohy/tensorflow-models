import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import tensorflow as tf

tf.reset_default_graph()

X = tf.placeholder(tf.float32, [None, 32, 32, 3])
y = tf.placeholder(tf.int64, [None])

def simple_model(X, y):
    Wconv1 = tf.get_variable('Wconv1', shape=[7, 7, 3, 32])
