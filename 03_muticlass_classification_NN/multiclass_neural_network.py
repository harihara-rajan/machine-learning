#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  4 10:55:17 2021

@author: hari-cms
"""
import numpy as np 
from scipy.io import loadmat
import matplotlib.pyplot as plt
from multiclass_classifier import *

# def sigmoid(z):
#     return (1/(1+np.exp(-z)))

def forward_pass(theta_1,theta_2,a1):
    """
    theta_1 - connects 1st i/p layer to hidden layer
    theta_2 - connects hidden layer to output layer
    a1 - activation at the input layer.
    """
    m = a1.shape[0]
    z_2 = a1 @ theta_1.T # (5000,25)
    # computing activation at a hidden layer 
    a2 = sigmoid(z_2) # a2 --> [a_1, a_2, a_3,..., a_25], shape (5000,25)
    ## need to add a_0=1 which is bias, refer lecture notes ##
    ones = np.ones(len(a2[:,0])).reshape(m,1)
    # stacking one to the computed activation
    a2 = np.hstack((ones,a2)) # 5000,26
    """
    output  computation
    -------------------
    z_3 = theta_2 * a2 
    h(x) = g(z_3)
    """
    # print(a2.shape)
    z_3 =  a2 @ theta_2.T
    a3 = sigmoid(z_3)
    return a3

if __name__ == "__main__":
    """
    learned parameters from the NN have been given to us. 
    And we have fetures given in "ex3data1.mat". we are building 
    3 layer NN to recogonise hand written digits.
    1st layer --> input layer 
    2nd layer --> hidden layer (25 units)
    3rd layer --> output layer (10 units) gives the result {0,1,..,9}

    """
    ## both weights and data are of dictionary type ##
    weights = loadmat("ex3weights.mat") # load data that contains learned params 
    data = loadmat("ex3data1.mat")
    X = data["X"]
    m = X.shape[0] # number of training exampe
    y = data["y"]
    y = y.reshape(m,1)
    theta_1 = weights["Theta1"] # connects 1st i/p layer to hidden layer
    theta_2 = weights["Theta2"] # connects hidden layer to output layer
    ## adding x0 feature ##
    ones = np.ones(len(X[:,0])).reshape(len(X[:,0]),1)
    # feature can be considered as an activation at input layer and hence
    # variable changed from x --> a
    x = np.hstack((ones,X))  
    lp = forward_pass(theta_1, theta_2, x)
    y_p = np.argmax(lp,axis=1)+1 # iterate row wise and throws index position of max values
    y_p = y_p.reshape(m,1)
    print("Training Set Accuracy:",sum(y_p==y)[0]/m*100,"%")