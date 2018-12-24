# -*- coding: utf-8 -*-
"""
Created on Thu Dec 20 13:29:06 2018

@author: ISHAN
"""
import numpy as np
import math
import matplotlib.pyplot as plt

data=np.loadtxt('LogisticRegressionData2.txt', delimiter=',') #dataset

m=118 #total no. of datasets
n=80 #no. of training examples used
n2=2 #no. of features

#making a function for feature scaling
def feat_scale(X,n,n2):
    f=np.random.rand(n,n2)
    for a in range(n2):
        f.T[a]=X.max(axis=0)[a]
    g=np.random.rand(n,n2)
    for a in range(n2):
        g.T[a]=X.min(axis=0)[a]
    h=np.random.rand(n,n2)
    for a in range(n2):
        h.T[a]=np.mean(X,axis=0)[a]
    X=np.divide((X-h),(f-g))
    return X;

#making the 'g' function
def g_func(x):
    g=1/(1+math.exp(-1*x))
    return g;

#making a function to carry out logistic regression problem and 
#giving accuracy of the program by entering the required details
def logistic_reg_grad_descent(data,m,n,n2):
    X=data[0:n,0:n2]
    X=feat_scale(X,n,n2)
    X=np.concatenate((np.ones((n,1)),X),axis=1) #because X0=1
    Y=data[0:n,n2]
    theta=np.random.rand(n2+1,1)*10
    iteration=1000 #no. of iterations
    p=1.5 #alpha value
    #For iteration
    theta1=theta #so that we can assign theta values simultaneously
    for i in range(iteration):
        for b in range(n2+1):
             s=0
             for a in range(n):
                 h=g_func(np.dot(X[a],theta))
                 s=s+(h-Y[a])*X[a,b]
             theta1[b]=theta[b]-p*s/n
        theta=theta1
    X1=data[n:,0:n2]
    X1=feat_scale(X1,m-n,n2)
    X1=np.concatenate((np.ones((m-n,1)),X1),axis=1) #because X0=1
    Y1=data[n:,n2]
    tx=np.dot(X1,theta)
    O=np.random.rand(m-n,1)
    #Assuming that if 'g' function gives a value >=0.5 then output 
    #is 1 else output is 0
    for a in range(m-n):
        if tx[a]>=0:
            O[a]=1
        else:
            O[a]=0
    #To calculate accuracy
    accuracy=np.random.rand(m-n,1)
    for k in range(m-n):
        if Y1[k]==O[k]:
            accuracy[k]=1
        else:
            accuracy[k]=0
    accuracy=np.sum(accuracy)
    accuracy=accuracy*100/(m-n)
    return accuracy;


accuracy=logistic_reg_grad_descent(data,m,n,n2)
print('accuracy' ,accuracy)

