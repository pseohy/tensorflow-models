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

def my_model(X, y, is_training):
 conv1 = tf.layers.conv2d(
        inputs=X,
        filters=64,
        kernel_size=[5,5],
        padding='same',
        activation=tf.nn.relu)

    batch_norm1 = tf.layers.batch_normalization(conv1, training=is_training)

    pool1 = tf.layers.max_pooling2d(
        inputs=batch_norm1,
        pool_size=[2,2],
        strides=2)

    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=128,
        kernel_size=[3,3],
        padding="same",
        activation=tf.nn.relu)

    batch_norm2 = tf.layers.batch_normalization(conv2, training=is_training)

    pool2 = tf.layers.max_pooling2d(
        inputs=batch_norm2,
        pool_size=[2,2],
        strides=2)

    conv3 = tf.layers.conv2d(
        inputs=pool2,
        filters=128,
        kernel_size=[3,3],
        padding="same",
        activation=tf.nn.relu)

    batch_norm3 = tf.layers.batch_normalization(conv3, training=is_training)
    
    conv33 = tf.layers.conv2d(
        inputs=batch_norm3,
        filters=128,
        kernel_size=[1,1],
        padding="same",
        activation=tf.nn.relu)
    
    batch_norm33 = tf.layers.batch_normalization(conv33, training=is_training)
    
    pool3 = tf.layers.max_pooling2d(
        inputs=batch_norm33,
        pool_size=[2,2],
        strides=2)

    pool3_flat = tf.reshape(pool3, [-1, 4 * 4 * 128])

    dense = tf.layers.dense(
        inputs=pool3_flat,
        units=1024,
        activation=tf.nn.relu)

    dropout = tf.layers.dropout(
        inputs=dense,
        rate=0.4,
        training=is_training)

    y_out = tf.layers.dense(inputs=dropout, units=10)

    return y_out

y_out  = my_model(X, y, is_training)

# loss
total_loss = tf.losses.softmax_cross_entropy(tf.one_hot(y, 10), logits=y_out)
mean_loss = tf.reduce_mean(total_loss)

# Adam optimizer with exponential learning rate decay
global_step = tf.Variable(0, trainable=False)
starter_learning_rate = 1e-3
learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step, 100000, 0.96, staircase=True)
optimizer = tf.train.AdamOptimizer(learning_rate)
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

