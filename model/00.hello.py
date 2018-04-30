import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import tensorflow as tf

hello = tf.constant('Hello, Tensorflow')

with tf.Session() as sess:
    hello_out = sess.run(hello)

print(hello_out)
