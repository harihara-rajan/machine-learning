import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd
import scipy.io


def display_data():
    """
    args
    ----
    X - gray scale immage data set 
    
    returns
    -------
    coverts the greay scale data and display it 
    """
    return None

def sigmoid(z):
    """
    return the sigmoid of z
    """
    
    return 1/ (1 + np.exp(-z))

def compute_cost(x,y,theta,Lambda):
    """
    args
    ----
    x - feature matrix, numpy array[m,n]
    m - number of training example 
    n - number of feature
    y - output/result, numpy array[m,1] 
    theta - parameter vector, numpy array[n,1]
    Lambda - regularization parameter

    returns
    -------
    regcost - regularized cost function value,scalar
    grad - gradient of regularized cost function wrt. n parameter, nparray[n,1]
    """

    m,n = x.shape # (118,28)
    grad = np.zeros((n,1))
    y_p = sigmoid(x@theta)
    error = (-y*np.log(y_p)) - ((1-y)*np.log(1-y_p))
    cost = (1/m) * np.sum(error)
    regcost = cost + Lambda/(2*m) * np.sum(theta[1:]**2)

    ## gradient computation ##
    diff = y_p - y # (118,1)
    j0 = 1/m * (diff.T @ x[:,0])  
    reg_term = (Lambda/m)*(theta[1:]) # (27,1)
    j1 = (1/m) * (x[:,1:].T @ diff ) + reg_term
    grad =  np.vstack((j0,j1)) # it stacks j0  and j1 vertically 
    return regcost,grad

def optimize(theta,x,y,lr,Lambda,num_iter):
    """
    args
    ----
    x - feature matrix, numpy array[m,n]
    m - number of training example 
    n - number feature, note the feture has been mapped from 2to28
    y - output/result, numpy array[m,1] 
    theta - parameter vector, numpy array[n,1]
    lr -  learning rate for Grad. Descent algorithm
    Lambda - regularization parameter 
    num_iter - maximum number of Grad. Descent iteration 

    returns 
    -------
    theta - learned parameter for the give data-set, np array[n,1]
    J - regularized cost function values for plotting purpose 
    """
    J = [] # cost function value
    I = [] # iteration 
    for i in range(num_iter):
        c,g = compute_cost(x,y,theta,Lambda) # c-> cost and g-> gradient
        if (i%100)==0:
            print(c)
        J.append(c)
        I.append(i)
        theta = theta - (lr*g)
    return theta,J,I



def one_vs_all(K,theta,x,y,lr,lmd,num_iter):
    """
    args
    ----
    K - 
    theta - parameter vector, np array[n,1]
    x - feature matrix, np array[m,n]
    y - output variable [m,1]
    lr - learning rate for gradient descent algorithm 
    lmd - lambda value 
    num_iter - number of grad. desc. iteration
    
    returns 
    -------
    learned_parameter matrix/vector depending on K 
    if K > 1, then learned_parameter will be a mat.
    else it will be a vector
    """
    learned_params = np.zeros((401,len(K)))    
    for i in range(len(K)):
        y_mod = np.zeros((5000,1))
        # temp varies from 0 to 9
        temp = K[i] # determines which among 10 classes needs to be trained
        
        ## In simple logistic regression, the value y should be with in {0,1}
        ## but since this is a multiclass classifier problem, the value of y is 
        ## from {0,9}. Therefore we cant perform logistic regression directly.
        ## to overcome this, I tried to make y to lie between {0,1} for which 
        ## I created y_mod initialised with zero's, changed the values to one's
        ## to only those position where y == temp 
        
        ## example ##
        #-----------#
        ## if temp is 9 
        ## in the original y, I found the position where y==9, once getting 
        ## posiion information, I simply changed the value of y_mod[pos]==1. 
        ## and now  I have a y_mod value that is between {0,1}
        
        pos = np.where(y==temp) #gives the position where y==temp
        y_mod[pos] = 1 # now y_mod->{0,1}
        lp,J,I = optimize(theta, x, y_mod, lr, lmd, num_iter)
        lp = lp.reshape(401,)
        learned_params[:,i] = lp
        plt.plot(I,J)
        plt.show()
    return learned_params
    
if __name__ == "__main__":
    mat = scipy.io.loadmat('ex3data1.mat')
    X = np.array(mat["X"])
    y = np.array(mat["y"])
    y[np.where(y==10)]=0
    K = [0,2,3]
    # need to add one to the first column of X
    ones = np.ones(len(X[:,0])).reshape(len(X[:,0]),1)
    x = np.hstack((ones,X))
    m,n = x.shape
    theta = np.zeros((n,1))
    lp = one_vs_all(K, theta, x, y, 0.01, 0.2, 300)
    print(lp)