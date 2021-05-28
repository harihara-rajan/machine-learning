import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 
import timeit
from scipy.optimize import minimize 

def feature_scaling(x):
    """
    Take in numpy array of X values and return normalize X values,
    the mean and standard deviation of each feature
    """
    mean=np.mean(X,axis=0)
    std=np.std(X,axis=0)
    X_norm = (X - mean)/std
    
    return X_norm , mean , std

def hypothesis(theta, x):
    """
    args
    ----
    theta - parameter vector, numpy array[n+1,1]
    n - number of features 
    x - feature matrix, numpy array[n+1,m]
    m - number of training examples 

    returns 
    -------
    hypothesis, which is a sigmoid function and 
    for a logistic regression 
    sigmoid(z) =       1
                  ------------
                  1  + exp (-z)
    and z = theta.T * X 
    """
    z = np.dot(theta.T,x) # np array [1,m]
    s = 1/(1+np.exp(-z))
    return s

def compute_cost(theta, x, y, hypo_vector):
    """
    args
    ---
    theta - parameter vector, numpy array[n+1,1]
    x - feature matrix, np array [n+1,m]
    y - output, np array [1,m]

    returns 
    -------
    cost function, scalar value 
    """
    m = len(y)
    ## Vectorised Implementation ##
    J = (np.sum(np.dot(-y,np.log(hypo_vector)) - np.dot((1-y),np.log(1-hypo_vector))))
    return J/m

def optimize(theta,x,y,lr,iter=50):
    """
    args
    ----
    theta - parameter vecctor,  np array [n+1, 1]
    n - number of features 
    x - feature matrix, np array [n+1,m]
    y - output, np array [1,m]
    lr- learning rate for gradient descent algorithm 
    iter - maximum number of grad. desc. iteration to be carried out

    returns 
    -------
    optimized parameter value that fits the data 
    """
    count = 0
    m = len(y) # number of training example
    cost = []
    co = []
    print("-"*80)
    print("Cost Function values")
    print("-"*80)
    while True:
        count += 1
        hypo_vector = hypothesis(theta,x)
        J = compute_cost(theta,x,y,hypo_vector)
        if count%100 == 0:
            print(J)
        cost.append(J)
        co.append(count)
        ## parameter update for next grad. desc. iteration ##
        dt0 = 1/m * np.sum(hypo_vector - y)
        dt1 = 1/m * np.sum(np.dot((hypo_vector - y),x[1,:]))
        dt2 = 1/m * np.sum(np.dot((hypo_vector-y),x[2,:]))
        t0 = theta[0] - lr*dt0
        t1 = theta[1] - lr*dt1
        t2 = theta[2] - lr*dt2
        theta = np.array([t0,t1,t2])
        ## breaking condition for while loop ##
        if count == iter:
            break 
    # plt.plot(co,cost)
    # plt.show()
    print("Gradient Descent completed at {} iteration".format(count))
    print('\n')
    return theta

def classifierPredict(optimized_parameter, X):
    """
    args
    ----
    optimized_parameters - learned parameters
    X - data set for which we need to predict if the 
        candidate with the mark stored in X can get 
        admission or not 
    returns 
    -------
    returns a boolean value. If True --> cadidate gets 
    admitted, if not, then cadidate did'nt get admission
    """

    predictions = np.dot(optimized_parameter.T,X)
    return predictions > 0


if __name__ == "__main__":
    start = timeit.default_timer()
    df = pd.read_csv("ex2data1.csv")
    X=df.iloc[:,:-1].values
    y=df.iloc[:,-1].values
    norm, mean, std = feature_scaling(X)

    x1 = np.array(df["x1"]) # feature 1 
    x1_s = (x1-np.mean(x1))/np.std(x1)
    x2 = np.array(df["x2"]) # feature 2
    x2_s = (x2-np.mean(x2))/np.std(x2)
    x0 = np.ones((x1.shape))
    x = np.array([x0,x1_s,x2_s])
    y = np.array((df["y"]))  # output vector 
    
    ## Data Visualisation ##
    pos = np.where(y==1)
    neg = np.where(y==0)
    
    ## start of parameter optimization process ##
    theta = np.array([0,0,0])
    optimized_parameter = optimize(theta,x,y,1,400)
    stop = timeit.default_timer()
    ## End of parameter optimization process ##

    ## Plotting decision boundary ##
    x_1 = np.array([np.max(x1_s),np.min(x1_s)])
    x_2 = (-1/optimized_parameter[2]) * (optimized_parameter[0] + optimized_parameter[1]*x_1)

    plt.scatter(x1_s[pos],x2_s[pos],label="Admitted",marker='x',c='g')
    plt.scatter(x1_s[neg],x2_s[neg],label="Not Admitted",marker='o', c='r')
    plt.plot(x_1, x_2, c='k')
    plt.xlabel("Exam 1 score")
    plt.ylabel("Exam 2 score")
    plt.legend()
    plt.show()

    ## prediction ## 
    student_mark_actual = np.array([90,50]).reshape(2,1)
    std = std.reshape(2,1)
    mean = mean.reshape(2,1)
    student_mark = (student_mark_actual - mean)/ std
    student_mark = np.append(np.ones(1), student_mark)
    
    ## prediction for a single student ## 
    prob_of_being_admitted = hypothesis(optimized_parameter,student_mark)
    prediction = classifierPredict(optimized_parameter,student_mark)
    print("-"*80)
    print("Prediction for the student with mark on subject x1 : {} and x2 : {}".format(student_mark_actual[0],student_mark_actual[1]))
    print("-"*80)
    print("Can the student get admission ? : {}".format(prediction))
    print("probability of student getting admission : {}".format(prob_of_being_admitted))
    ##  another way to evaluate the quality of parameters that we found 
    ##  is to see how well  the learned model predicts our training ex. 
    print("\n")
    print("-"*80)
    print("Evaluating the Quality of parameters found")
    print("-"*80)
    p=classifierPredict(optimized_parameter,x)
    p = p.astype('uint8') # converts boolean to int with 0's and 1's
    print("Train Accuracy:", sum(p==y),"%")
    print("Time taken : {} seconds".format(stop-start))
    print('\n')