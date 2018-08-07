from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import seaborn as sns
import scipy
import keras
from scipy.interpolate import interp1d
from scipy.stats import zscore
import os
from sklearn import metrics
from tensorflow.examples.tutorials.mnist import input_data
import random
import copy
import itertools

n_sample_train = 10000
n_sample_test = 1000

def normalize_data(data):

    std = []
    mean = []
    dataOut = copy.copy(data)


    for i in range(dataOut.shape[0]):

        std.append([np.std(dataOut, ddof=1) for j in range(dataOut.shape[0])])
        mean.append([np.mean(dataOut) for k in range(dataOut.shape[0])])

    std = list(itertools.chain(*std))
    mean = list(itertools.chain(*mean))

    for i in range(data.shape[0]):
        dataOut[i, :] = (dataOut[i, :] - mean[i])/ std[i]

        dataOut[np.isnan(dataOut)] = 0

    return dataOut

def interp_data(data,length):

    new_dataOut = np.zeros((length, data.shape[1]))
    for i in range(data.shape[1]):
        temp = data[:, i]
        x = np.linspace(0, 1, data.shape[0])
        x_new = np.linspace(0, 1, length)
        new_data = interp1d(x, temp)(x_new) #error
        new_dataOut[:, i] = new_data

    return new_dataOut

#get_WIIBB
def get_WIIBB_data():
    subjects = ['akira','harada','kazushi']
    movementNames = ['a','b','c','d','e']
    cwd = os.getcwd() #get carrent director
    pathBase = cwd + '\\motion\\'

    train_data = np.zeros((3, 25, 200, 6))

    #import data
    i = 0
    j = 0

    for subject in subjects:
        pathSub = pathBase + subject + '\\'
        files = os.listdir(pathSub)

        for movementName in movementNames:
            subFiles = [s for s in files if s.startswith(movementName + '_')]
            subFiles.sort()

            for k in range(len(subFiles)):
                print(subject, movementName, subFiles[k], 'file name')
                dataPath = pathSub + subFiles[i]
                data = np.loadtxt(dataPath, dtype='float', delimiter=',')
                data = np.delete(data, 0, 1) #dlete time
                print('data shape', data.shape)
                data_200 = interp_data(data, 200)
                train_data[i, j, :, :] = data_200
            j = j + 1
        i = i + 1
    n_data_200 = normalize_data(train_data)


    train_x = n_data_200[[0,1], :, :, :]
    test_x = n_data_200[[2], :, :, :]

    train_x = train_x.reshape(-1, 6, 200,1)
    test_x = test_x.reshape(-1, 6, 200,1)
    train_x = train_x[:50,:,:,:]
    test_x = test_x[:,:,:,:]

    one_hots_train = np.zeros((50,5))
    one_hots_train[(0, 1, 2, 3, 4), 0] = 1
    one_hots_train[(5, 6, 7, 8, 9), 1] = 1
    one_hots_train[(10, 11, 12, 13, 14), 2] = 1
    one_hots_train[(15, 16, 17, 18, 19), 3] = 1
    one_hots_train[(20, 21, 22, 23, 24), 4] = 1
    one_hots_train[(25, 26, 27, 28, 29), 0] = 1
    one_hots_train[(30, 31, 32, 33, 34), 1] = 1
    one_hots_train[(35, 36, 37, 38, 39), 2] = 1
    one_hots_train[(40, 41, 42, 43, 44), 3] = 1
    one_hots_train[(45, 46, 47, 48, 49), 4] = 1

    one_hots_test = np.zeros((25,5))
    one_hots_test[(0, 1, 2, 3, 4), 0] = 1
    one_hots_test[(5, 6, 7, 8, 9), 1] = 1
    one_hots_test[(10, 11, 12, 13, 14), 2] = 1
    one_hots_test[(15, 16, 17, 18, 19), 3] = 1
    one_hots_test[(20, 21, 22, 23, 24), 4] = 1

    return train_x, one_hots_train, test_x, one_hots_test

def dense(input, name, in_size, out_size, activation="relu"):

    with tf.variable_scope(name, reuse=False):
        w = tf.get_variable("w", shape=[in_size, out_size],
                            initializer=tf.random_normal_initializer(mean=0, stddev=0.1))
        b = tf.get_variable("b", shape=[out_size], initializer=tf.constant_initializer(0.0))

        l = tf.add(tf.matmul(input, w), b)

        if activation == "relu":
            l = tf.nn.relu(l)
        elif activation == "sigmoid":
            l = tf.nn.sigmoid(l)
        elif activation == "tanh":
            l = tf.nn.tanh(l)
        else:
            l = l
        print(l)
    return l

def scope(y, y_, learning_rate=0.1):

    #Learning rate
    learning_rate = tf.Variable(learning_rate,  trainable=False)

    # Loss function
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        labels=y, logits=y_), name="loss")

    # Optimizer
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate,
                                       name="optimizer").minimize(loss)

    # Evaluate the model
    correct = tf.equal(tf.cast(tf.argmax(y_, 1), tf.int32),
                       tf.cast(tf.argmax(y, 1), tf.int32))
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32), name="accuracy")

    #  Tensorboard
    writer = tf.summary.FileWriter('./Tensorboard/')
    # run this command in the terminal to launch tensorboard:
    # tensorboard --logdir=./Tensorboard/
    writer.add_graph(graph=sess.graph)

    return loss, accuracy, optimizer, writer

def confusion_matrix(cm, accuracy):

    plt.figure(figsize=(9, 9))
    sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square=True, cmap='Blues_r')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    all_sample_title = 'Accuracy Score: {0}'.format(accuracy)
    plt.title(all_sample_title, size=15)

#main
train_x, one_hots_train, test_x, one_hots_test = get_WIIBB_data() #get train and test data
number_test = [one_hots_test[i, :].argmax() for i in range(0, one_hots_test.shape[0])]

#plot_MNIST(x=train_x, one_hot=one_hots_train)

n_label = len(np.unique(number_test))   # Number of class
height = train_x.shape[1]
width = train_x.shape[2]

# Session and context manager
tf.reset_default_graph()
sess = tf.Session()

with tf.variable_scope(tf.get_variable_scope()):

    # Placeholders
    x = tf.placeholder(tf.float32, [None, height, width, 1], name='X')
    y = tf.placeholder(tf.float32, [None, n_label], name='Y')

    keep_prob = tf.placeholder(tf.float32, name='keep_prob')

    # Convolutional Neural network
    c1 = tf.layers.conv2d(inputs=x, kernel_size=[1, 5], strides=[1, 1],
                          filters=16, padding='SAME', activation=tf.nn.relu,
                          name='Conv_1')
    print(c1) #error
    c1 = tf.layers.max_pooling2d(inputs=c1, pool_size=[2, 2],
                                 strides=[2, 2],padding='SAME')
    print(c1)

    c2 = tf.layers.conv2d(inputs=c1, kernel_size=[3, 3], strides=[1, 1],
                          filters=32, padding='SAME', activation=tf.nn.relu,
                          name='Conv_2')
    print(c2)
    c2 = tf.layers.max_pooling2d(inputs=c2, pool_size=[2, 2],
                                 strides=[2, 2], padding='SAME')
    print(c2)

    # Reshape to a fully connected layers
    size = c2.get_shape().as_list()

    l1 = tf.reshape(c2, [-1, size[1] * size[2] * size[3]], name='reshape_to_fully')
    print(l1)

    l2 = dense(input=l1, name="output_layers", in_size=l1.get_shape().as_list()[1], out_size=n_label, activation='None')


    # Softmax layer
    y_ = tf.nn.softmax(l2, name='softmax')
    print(y_)

    # Scope
    loss, accuracy, optimizer, writer = scope(y, y_, learning_rate=0.00005)

    # Initialize the Neural Network
    sess.run(tf.global_variables_initializer())

    # Train the Neural Network
    loss_history = []
    acc_history = []
    epoch = 500
    train_data = {x: train_x, y: one_hots_train}

    for e in range(epoch):

        _, l, acc = sess.run([optimizer, loss, accuracy], feed_dict=train_data) #error

        loss_history.append(l)
        acc_history.append(acc)

        print("Epoch " + str(e) + " - Loss: " + str(l) + " - " + str(acc))

plt.figure()
plt.plot(acc_history)

# Test the trained Neural Network
test_data = {x: test_x, y: one_hots_test}
l, acc = sess.run([loss, accuracy], feed_dict=test_data)
print("Test - Loss: " + str(l) + " - " + str(acc))
predictions = y_.eval(feed_dict=test_data, session=sess)
predictions_int = (predictions == predictions.max(axis=1, keepdims=True)).astype(int)
predictions_numbers = [predictions_int[i, :].argmax() for i in range(0, predictions_int.shape[0])]
# Confusion matrix
cm = metrics.confusion_matrix(number_test, predictions_numbers)
print(cm)
confusion_matrix(cm=cm, accuracy=acc)
#cmN = cm / cm.sum(axis=0)
#confusion_matrix(cm=cmN, accuracy=acc)
