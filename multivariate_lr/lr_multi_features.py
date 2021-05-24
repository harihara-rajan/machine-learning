import numpy as np
from numpy.core.fromnumeric import mean
import pandas as pd 
import matplotlib.pyplot as plt 

def hypothesis(theta, x):
    """
    args
    ----
    theta - parameter vector 
    x - feature matrix with each row has a specific feature value of m training ex.

    returns 
    -------
    hypothesis for multivariate regression model 
    h_theta(x0,x1,...,xn) = theta_0*x0 + theta_1*x1 + .... theta_n*xn
    """
    # either of the hypo can be choosed both are equivalent
    # hypo = theta[0]*x[0,:]+ theta[1,:]*x1 + theta[2,:]*x2
    theta = theta.reshape(3,1)
    hypo = np.dot(theta.T,x)
    return hypo

def cost_function(hypothesis,y):
    """
    args
    ----
    hypothesis : hypothesis of multivatiate regression,numpy[1,m]
    m : nummber of training example 

    returns
    -------
    cost function value, scalar value 
    """
    m = len(y) # number of training example 
    ## cost function of linear regression ##
    J = 1/(2*m) * np.sum(np.dot((hypothesis - y).T, (hypothesis - y))) 
    return J 

def optimize(theta,lr,x,y,iter):
    """
    args
    ---
    theta : parameter vector, numpy array[n+1,1]
    n : nummber of features 
    lr : learning rate for gradient descent algorithm 
    x : feature matrix, numpy[n+1,m]
    m : numbe rof training example 
    iter: number of gradient descent iteration required 

    returns 
    -------
    theta : learned parameter vector for a given data-set, numpy array[n+1,1]
    """
    m = len(y)
    count = 0
    cost = []
    iteration = []
    ## start of gradient descent algorithm ##
    while True:
        count += 1 
        ## computing hypothesis for a given parameter vector and feature matrix ##
        hypo = hypothesis(theta,x) 
        ## computing cost function ##
        J = cost_function(hypo,y)
        ## for plotting J vs number of iteration ## 
        iteration.append(count)
        cost.append(J)
        ## printing cost function for every 10 iteration ##
        if count%10 == 0:
            print(J)
        
        dt0 = 1/m * np.sum(hypo - y)
        dt1 = 1/m * np.sum(np.dot((hypo - y),x[1,:]))
        dt2 = 1/m * np.sum(np.dot((hypo-y),x[2,:]))
        t0 = theta[0] - lr * dt0
        t1 = theta[1] - lr * dt1
        t2 = theta[2] - lr * dt2
        theta = np.array([t0,t1,t2])
        if count == iter:
            break
    plt.plot(iteration, cost)
    plt.show()
    return theta

if __name__ == "__main__" :
    # print("name")
    df = pd.read_csv("ex1data2.csv")
    ## extracting feature value ##
    x1_no_scaled = np.array((df["x1"]))
    x2_no_scaled = np.array((df["x2"]))
    ## scaling the feature values for good performance of algorithm ##
    x1 = (x1_no_scaled - mean(x1_no_scaled))/np.std(x1_no_scaled)
    x2 = (x2_no_scaled - mean(x2_no_scaled))/np.std(x2_no_scaled)
    x0 = np.ones((x2.shape))
    ## creating a feature matrix ##
    x = np.array([x0,x1,x2])
    ## output values ##
    y = np.array((df["y"])) 
    ## initial guess for parameters to start the grad. desc. algorithm
    theta_initial = np.array([0,0,0])
    t_array = optimize(theta_initial,0.1,x,y,iter= 250) # gradient descent function call 
    print(t_array)

    ## predicting the house price with 1650 area and 3 bed-rooms 
    area_scaled = (1650-np.mean(x1_no_scaled))/np.std(x1_no_scaled)
    bedroom_scaled = (3-mean(x2_no_scaled))/np.std(x2_no_scaled)
    price = t_array[0] + area_scaled*t_array[1] + bedroom_scaled*t_array[2]
    print(price)
