# Simple cnn implementation with solver

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import tensorflow as tf
import numpy as np
import math

tf.reset_default_graph()

X_train = np.random.rand(1000, 32, 32, 3)
y_train = np.random.randint(0, 10, (1000,))
X_val= np.random.rand(50, 32, 32, 3)
y_val = np.random.randint(0, 10, (50,))

X = tf.placeholder(tf.float32, [None, 32, 32, 3])
y = tf.placeholder(tf.int64, [None])
is_training = tf.placeholder(tf.bool)

def simple_model(X, y):
    Wconv1 = tf.get_variable('Wconv1', shape=[7, 7, 3, 32])
    bconv1 = tf.get_variable('bconv1', shape=[32])
    W1 = tf.get_variable('W1', shape=[5408, 10])
    b1 = tf.get_variable('b1', shape=[10])

    a1 = tf.nn.conv2d(X, Wconv1, strides=[1,2,2,1], padding='VALID')
    h1 = tf.nn.relu(a1)
    h1_flat = tf.reshape(h1, [-1, 5408])
    y_out = tf.matmul(h1_flat, W1) + b1
    return y_out

y_out = simple_model(X, y)

total_loss = tf.losses.hinge_loss(tf.one_hot(y, 10), logits=y_out)
mean_loss = tf.reduce_mean(total_loss)

optimizer = tf.train.AdamOptimizer(5e-4)
train_step = optimizer.minimize(mean_loss)

def run_model(session, predict, loss_val, Xd, yd,
              epochs=1, batch_size=64,
              training=None, plot_losses=False):
    correct_prediction = tf.equal(tf.argmax(predict, 1), y)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    train_indices = np.arange(Xd.shape[0])
    np.random.shuffle(train_indices)

    training_now = training is not None

    # setting up variables we want to compute (and optimize)
    variables = [mean_loss, correct_prediction, accuracy]
    if training_now:
        variables[-1] = training

    # counter
    iter_cnt = 0
    for e in range(epochs):
        correct = 0
        losses = []

        for i in range(int(math.ceil(Xd.shape[0] / batch_size))):
            start_idx = (i * batch_size) % Xd.shape[0]
            idx = train_indices[start_idx: start_idx + batch_size]

            # create a feed dictionary for this batch
            feed_dict = { X: Xd[idx, :],
                          y: yd[idx],
                          is_training: training_now }

            actual_batch_size = yd[idx].shape[0]

            loss, corr, _ = sess.run(variables, feed_dict=feed_dict)

            losses.append(loss * actual_batch_size)
            correct += np.sum(corr)

        total_correct = correct / Xd.shape[0]
        total_loss = np.sum(losses) / Xd.shape[0]
        print('Epoch {2}, Overall loss = {0:.3g} and accuracy of {1:.3g}'\
              .format(total_loss, total_correct, e + 1))

    return total_loss, total_correct

with tf.Session() as sess:
    with tf.device('/cpu:0'):
        sess.run(tf.global_variables_initializer())
        run_model(sess, y_out, mean_loss, X_train, y_train, 10, 64,
                  train_step, True)
        run_model(sess, y_out, mean_loss, X_val, y_val, 1, 64)
