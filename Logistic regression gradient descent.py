# -*- coding: utf-8 -*-
"""
Created on Thu Dec 20 13:29:06 2018

@author: ISHAN
"""
import numpy as np
import math
import matplotlib.pyplot as plt

data=np.loadtxt('LogisticRegressionData2.txt', delimiter=',') #importing dataset
dataset=118 #total no. of datasets
training_example=80 #no. of training examples used
feature=2 #no. of features

#making a function for feature scaling
def feat_scale(X,training_example,feature):
    Xmax=np.random.rand(training_example,feature)
    for a in range(feature):
        Xmax.T[a]=X.max(axis=0)[a]
    Xmin=np.random.rand(training_example,feature)
    for a in range(feature):
        Xmin.T[a]=X.min(axis=0)[a]
    Xmean=np.random.rand(training_example,feature)
    for a in range(feature):
        Xmean.T[a]=np.mean(X,axis=0)[a]
    X=np.divide((X-Xmean),(Xmax-Xmin))
    return X;

#making the 'g' function
def g_func(x):
    g=1/(1+math.exp(-1*x))
    return g;

#making a function to carry out logistic regression problem and 
#giving accuracy of the program by entering the required details
def logistic_reg_grad_descent(data,dataset,training_example,feature):
    X=data[0:training_example,0:feature]
    X=feat_scale(X,training_example,feature)
    X=np.concatenate((np.ones((training_example,1)),X),axis=1) #because X0=1
    Y=data[0:training_example,feature]
    theta=np.random.rand(feature+1,1)*10
    iteration=1000 #no. of iterations
    alpha=1.5 #alpha value
    #For iteration
    theta1=theta #so that we can assign theta values simultaneously
    for i in range(iteration):
        for b in range(feature+1):
             cost_func=0
             for a in range(training_example):
                 hypo=g_func(np.dot(X[a],theta))
                 cost_func=cost_func+(hypo-Y[a])*X[a,b]
             theta1[b]=theta[b]-alpha*cost_func/training_example
        theta=theta1
    X1=data[training_example:,0:feature]
    X1=feat_scale(X1,dataset-training_example,feature)
    X1=np.concatenate((np.ones((dataset-training_example,1)),X1),axis=1) #because X0=1
    Y1=data[training_example:,feature]
    thetaX=np.dot(X1,theta)
    output=np.random.rand(dataset-training_example,1)
    #Assuming that if 'g' function gives a value >=0.5 then output 
    #is 1 else output is 0
    for a in range(dataset-training_example):
        if thetaX[a]>=0:
            output[a]=1
        else:
            output[a]=0
    print('Output is:' ,output)
    #To calculate accuracy
    accuracy=np.random.rand(dataset-training_example,1)
    for k in range(dataset-training_example):
        if Y1[k]==output[k]:
            accuracy[k]=1
        else:
            accuracy[k]=0
    accuracy=np.sum(accuracy)
    accuracy=accuracy*100/(dataset-training_example)
    return accuracy;

accuracy=logistic_reg_grad_descent(data,dataset,training_example,feature)
print('accuracy' ,accuracy)

