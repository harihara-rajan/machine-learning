import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 

def mapFeature(x1,x2,degree):
    """
    description
    ------------
    take in numpy array of x1 and x2, return all polynomial 
    terms up to the given degree
    Reference 
    ---------
    This part of code has been adapted from "Benlau93" github user
    link : https://github.com/Benlau93/Machine-Learning-by-Andrew-Ng-in-Python
    blog : https://towardsdatascience.com/andrew-ngs-machine-learning-course-in-python-regularized-logistic-regression-lasso-regression-721f311130fb
    """
    out = np.ones(len(x1)).reshape(len(x1),1)
    for i in range(1,degree+1):
        for j in range(i+1):
            terms= (x1**(i-j) * x2**j).reshape(len(x1),1)
            out= np.hstack((out,terms))
    return out

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
    n - number feature, note the feture has been mapped from 2to28
    y - output/result, numpy array[m,1] 
    theta - parameter vector, numpy array[n,1]
    Lambda - regularization parameter

    returns
    -------
    regcost - regularized cost function value,scalar
    grad - gradient of regularized cost function wrt. n parameter, numpy array[n,1]
    """

    m,n = x.shape # (118,28)
    grad = np.zeros((n,1))
    y_p = sigmoid(x@theta)
    cost = (1/m) * np.sum(-y*np.log(y_p) - (1-y)*np.log(1-y_p))
    regcost = cost + Lambda/(2*m) * np.sum(theta**2)

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

def predict(x,learned_theta):
    """
    args
    ----
    x - training data set
    y - actual output result  
    theta - learned parameters

    return
    ------
    pred - prediction value 

    description 
    -----------
    note that in logistic regression, we have a 
    threshhold value for a sigmoid function. if 
    the sigmoid function predicts a value greater
    than 0.5 for a particular sample, then the 
    particular sample will be treated as a positive
    one, (i.e) pred = 1 if not pred = 0   
    """
    z = x@learned_theta
    y_p = sigmoid(z)
    pred = np.zeros((y_p.shape))
    for i in range(len(y_p)):
        if y_p[i] >= 0.5:
            pred[i] = 1
        else:
            pred[i] = 0
    return pred
if __name__ == "__main__":
    df = pd.read_csv("ex2data2.csv", header=None)
    X = np.array(df.iloc[:,0:2]) # feature matrix 
    Y = np.array(df.iloc[:,2])   # output 
    pos = np.where(Y==1)
    neg = np.where(Y==0) 
    x = mapFeature(X[:,0], X[:,1],6)  # (118,28)
    m,n= x.shape
    y = Y.reshape(m,1)
    initial_theta = np.zeros((n, 1))  #   (28,1)
    params, J_vals, I_vals  = optimize(initial_theta,x,y,1,0.2,800)
    ## plot and check how the function value decreases for every iteration ##
    # plt.plot(I_vals,J_vals,c='m')
    # plt.xlabel("Iteration ")
    # plt.ylabel("Cost function $J(\Theta)$")
    
    ## plot and check the data ##
    # plt.scatter(X[:,0][pos],X[:,1][pos],c='g', label="y = 1")
    # plt.scatter(X[:,0][neg],X[:,1][neg],c='r',marker='+', label = "y=0")
    # plt.legend()
    # plt.show()

    ## accuracy of the model ##
    """
    one way to predict the accuracy of the trained model is by 
    passing the training data set and the learned parameter to 
    predict function and check how may training examples are pred.
    correctly 
    """
    prediction = predict(x,params)
    # print(sum(prediction==y))
    print("The training accuracy is ",sum(y==prediction)/len(y)*100,"%" )