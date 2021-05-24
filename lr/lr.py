import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 

def hypothesis(slope,intercept,x):
    """
    args:
    ----
    x - numpy array of training example
    slope - slope of regression line 
    intercpet - intercept of regression line 
    
    returns:
    --------
    hypothesis/predicted value for a given slope,inter, and x.
    hypothesis has same length of x 
    """
    # print(intercept + slope*x)
    return intercept + slope*x

def compute_cost(hypothesis, y):
    """
    args
    ----
    hypothesis - it is the predicted value obtained from hypothesis function
                 numpy array 
    y - actual value from the training example
        numpy array 
    
    returns
    -------
    cost function - a scalar value 
    """
    m = len(y) # number of training example 
    J = 1/(2*m) * np.sum( (hypothesis - y)**2) # cost function
    return J 

def optimize(guess_slope, guess_intercept, lr,x,y, no_gd_iter):
    """
    args
    ----
    guess_slope - initial guess of slope for gradient descent algo. 
    guess_intercept - intitial guess of intercept for gradient descent algo
    lr - learning rate 
    x - independent variable, numpy array of length m
    y - dependent variable , numpy array of length m
    no_gd_iter - number of gradient descent iteration 

    returns
    -------
    predicated slope and intercept 
    """
    m = len(y)
    count = 0 
    while True :
        count += 1 
        hypo_vector = hypothesis(guess_slope,guess_intercept,x)
        print(hypo_vector.shape)
        J = compute_cost(hypo_vector,y)
        # if (count%100 == 0):
        #     print(J)
        ds = 1/m * np.sum(np.dot((hypo_vector-y),x)) # derivative of cost function with respect to slope 
        di = 1/m * np.sum(hypo_vector - y) # derivative of the cost function with respect to intercept  
        print(di)
        guess_slope = guess_slope - lr*ds # predicted slope for next iteration 
        guess_intercept = guess_intercept - lr*di # predicted intercept for next iteration
        if (count == 6000):
            break
    pred_slope = guess_slope
    pred_inter = guess_intercept
    return pred_inter,pred_slope

if __name__ == "__main__":
    df = pd.read_csv("ex1data1.csv") # reading the csv data set 
    x = df["x"] # independent variable 
    y = df["y"] # dependent variable 

    i,s = optimize(0,0,0.009, x ,y , 5000) # function call; s-> slope, i->intercept
    # print("intercept : {} and slope : {}".format(i,s)) # printing s,i to conoole
    y_p = hypothesis(s,i,x) # predicting the dependent variable 

    # plotting for visualisations
    plt.scatter(x,y,marker="x",c='r',label="training data") 
    plt.plot(x,y_p,c="b", label="linear regression")
    plt.legend()
    plt.show()