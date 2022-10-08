#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 29 13:08:33 2022

@author: liuyilouise.xu
"""
from scipy.special import expit # vectorized sigmoid, need for mu0.5, batch500
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import math
import matplotlib.pyplot as plt
import os

data = pd.read_csv("letter-recognition.data", header = None)
labels = data[0]
data = data[np.arange(1,17)]

X_train, X_test, Y_train, Y_test  = train_test_split(data, labels, test_size = 0.2) 

X_train = X_train.values
X_test = X_test.values
Y_train = Y_train.values - 1 #label ranges from 1-26, so minus 1 to change to 0-25
Y_test = Y_test.values - 1

X_train = (X_train -  X_train.min(0)) / X_train.ptp(0) # normalize each column
X_test = (X_test -  X_test.min(0)) / X_test.ptp(0) # normalize each column


def sigmoid (x):
    return 1/(1+math.exp(-x))

sigmoid_v = np.vectorize(sigmoid)

def sigmoid_dev(y):
    return y*(1-y)

sigmoid_dev_v = np.vectorize(sigmoid_dev)

def softmax (vec):
    return pow(math.e, vec) / sum(pow(math.e, vec))

# train on a batch of data, return new weights
def ann_train(batch, label, mu, wih1, wh1h2, wh2o):
    rows = batch.shape[0]

    batch = np.append(np.ones((rows,1)), batch , axis=1) # append bias
    yh1 = np.dot(batch, wih1) 
    yh1 = expit(yh1)
    #yh1 = sigmoid_v(yh1) # yh1 outputs for batch
    
    yh1 = np.append(np.ones((rows,1)), yh1 , axis=1)
    yh2 = np.dot(yh1, wh1h2) 
    yh2 = expit(yh2)
    #yh2 = sigmoid_v(yh2) # yh2 outputs for batch
    
    # 10 * 27 = batch size * num of H2 neorons + 1
    yh2 = np.append(np.ones((rows,1)), yh2 , axis=1)
    yo = np.dot(yh2, wh2o) 
    yo = expit(yo)
    #yo = sigmoid_v(yo) # yo outputs for batch
    
    
    # construct target matrix, 0.9 for correct label, 0.1 otherwise
    target = np.full((batch_size, 26), 0.1)
    for row, lab in enumerate(label):
        target[row, lab] = 0.9
    
    # layer O - H2
    # 10*26 = batch size * num output neurons
    error = (yo - target) 
    delta_o = error # E / out_o = error
    # 27*26 = num of H2 neorons + 1 * num output neurons
    adjust_wh2o = np.dot(yh2.transpose(), delta_o) # E / w_h2o
    wh2o_new = wh2o - mu*adjust_wh2o
    
    # layer H2 - H1
    # 26*26 = num of H2 neorons * num output neurons
    wh2o_t = np.delete(wh2o, 0, axis=0) #temporarily delet H20 row because not needed in weight adjust
    # 10*26 = batch size * num H2 neurons
    delta_wh1h2 = np.dot(delta_o, wh2o_t.T) # E / out_H2
    # 10*26 = batch size * num H2 neurons
    yh2_t = np.delete(yh2, 0, axis=1) # temporarily delet yH20 column
    delta_sigma_h2 = np.apply_along_axis(sigmoid_dev_v,1, yh2_t) # out_H2 / in_H2
    # 10*26 = batch size * num H2 neurons
    delta_h2 = np.multiply(delta_wh1h2, delta_sigma_h2) # E / in_H2, element wise multiply
    
    # layer H1 - In
    
    wh1h2_t = np.delete(wh1h2, 0, axis=0) #temporarily delet H10 row because not needed in weight adjust
    delta_wih1 = np.dot(delta_h2, wh1h2_t.T) # E / out_H1
    yh1_t = np.delete(yh1, 0, axis=1)
    activation_dev_h1 = np.apply_along_axis(sigmoid_dev_v,1, yh1_t) # out_H1 / in_H1
    delta_h1 = np.multiply(delta_wih1,activation_dev_h1) # E / in_H1
    
    # adjust weight for each batch data
    for i in np.arange(rows):
        adjust_wh1h2 = np.outer(yh1[i,:], delta_h2[i,:]) # E / w_h1h2
        wh1h2_new = wh1h2 - mu*adjust_wh1h2
        
        adjust_wih1 = np.outer(batch[i,:], delta_h1[i,:]) # E / w_ih1
        wih1_new = wih1 - mu*adjust_wih1
    
    training_error = np.sum(np.true_divide(np.power(error, 2),batch_size))
    
    return wih1_new, wh1h2_new, wh2o_new, training_error

def ann_validate(batch, label, wih1, wh1h2, wh2o, buildCM=False):
    rows = batch.shape[0]
    
    batch = np.append(np.ones((rows,1)), batch , axis=1) # append bias
    yh1 = np.dot(batch, wih1) 
    yh1 = expit(yh1)
    #yh1 = sigmoid_v(yh1) # yh1 outputs for batch
    
    yh1 = np.append(np.ones((rows,1)), yh1 , axis=1)
    yh2 = np.dot(yh1, wh1h2) 
    yh2 = expit(yh2)
    #yh2 = sigmoid_v(yh2) # yh2 outputs for batch
    
    yh2 = np.append(np.ones((rows,1)), yh2 , axis=1)
    yo = np.dot(yh2, wh2o) 
    yo = expit(yo)
    #yo = sigmoid_v(yo) # yo outputs for batch
    
    out = np.apply_along_axis(softmax, 1, yo) # apply softmax to each row of outputs
    
    pred = np.argmax(out, axis=1)
    #pred = np.asarray(out == np.max(out, axis=1, keepdims=True)).nonzero()[1] # index with max elt
    
    accuracy= np.mean(pred == label)
    
    if buildCM:
        CM = np.zeros(shape = (26, 26))
        for i in np.arange(pred.shape[0]):
            CM[pred[i]][label[i]] += 1
    else:
        CM = 0
        
    return accuracy, CM
'''
# test for XOR
X_train = np.array([[1,1],[1,0],[0,1], [0,0],[1,1],[1,0],[0,1], [0,0],[1,1],[1,0],[0,1], [0,0]])
Y_train = np.array([0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1 ])

X_test = np.array([[1,1],[1,0],[0,1], [0,0],[1,1],[1,0],[0,1], [0,0],[1,1],[1,0],[0,1], [0,0]])
Y_test = np.array([0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0 ])
'''

def train(data, label, batch_size, mu, n_epoch, wih1, wh1h2, wh2o):
    rows = data.shape[0]
    accuracies = []
    train_error = []
    
    for i in np.arange(n_epoch):
        batch_idx = np.random.randint(rows, size=batch_size) 
        batch_data = data[batch_idx]
        batch_label = label[batch_idx]
        
        wih1, wh1h2, wh2o, training_error = ann_train(batch_data, batch_label, mu, wih1, wh1h2, wh2o)
        
        if (i == (n_epoch-1)):
            '''
            file = open(f'batch{batch_size}_mu{mu}_wih1', "w")
            for row in wih1:
                np.savetxt(file, row)
            file = open(f'batch{batch_size}_mu{mu}_wh1h2', "w")
            for row in wh1h2:
                np.savetxt(file, row)
            file = open(f'batch{batch_size}_mu{mu}_wh2o', "w")
            for row in wh2o:
                np.savetxt(file, row)
            '''
            accuracy, CM = ann_validate(X_test, Y_test, wih1, wh1h2, wh2o, buildCM=True)

        else:
            accuracy, CM = ann_validate(X_test, Y_test, wih1, wh1h2, wh2o)

        accuracies.append(accuracy)
        train_error.append(training_error)
        
    return accuracies, train_error, CM

'''
# test for XOR
wih1 = np.random.uniform(-1, 1, 3*26).reshape((3, 26))
wh1h2 = np.random.uniform(-1, 1, 27*26).reshape((27, 26))
wh2o = np.random.uniform(-1, 1, 27*2).reshape((27, 2))

batch_size = 3 #1, 10, 100, and 500
mu = 0.1 #0.5, 0.1, 0.01 and 0.001 
n_epoch = 1000


accuracies, train_error, CM = train(X_train, Y_train, batch_size, mu, n_epoch, wih1, wh1h2, wh2o)
'''

n_epoch = 10000
batch_sizes = [1, 10, 100, 500]
mus = [0.1, 0.01, 0.001, 0.5]

# save confusion matrices to directory CMs
os.makedirs("./CMs")
for mu in mus:
    plt.title("Accuracy, mu = " + str(mu))
    for batch_size in batch_sizes:
        wih1 = np.random.uniform(-1, 1, 17*26).reshape((17, 26))
        wh1h2 = np.random.uniform(-1, 1, 27*26).reshape((27, 26))
        wh2o = np.random.uniform(-1, 1, 27*26).reshape((27, 26))
        
        accuracies, train_error, CM = train(X_train, Y_train, batch_size, mu, n_epoch, wih1, wh1h2, wh2o)
        np.savetxt("./CMs/CM_batchsize"+str(batch_size) + "_mu"+ str(mu)+ "_epoch" + str(n_epoch)+".txt", CM, fmt='%d')
        plt.plot(accuracies, label = "%d %f"%(batch_size, mu))
        print("Complete one batch condition")
            
    plt.legend(loc='upper center')
    plt.show()

# save gray-scal matrices to directory CMs_gray
os.makedirs("./CMs_gray")
for filename in os.listdir("./CMs"):
    batchstr, mustr = filename.split("_")[1:3]
    #plt.gray()
    plt.title("Confusion matrix, " + batchstr +" " + mustr)
    CM = np.loadtxt("./CMs/" + filename)
    plt.imshow(CM, cmap='gray')
    plt.savefig("./CMs_gray/" + filename.replace(".txt",".jpg"))
    plt.show()


