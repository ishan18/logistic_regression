# -*- coding: utf-8 -*-
"""
Created on Thu Dec 20 13:29:06 2018

@author: ISHAN
"""
import numpy as np
import math
import matplotlib.pyplot as plt

data=np.loadtxt('LogisticRegressionData2.txt', delimiter=',')

m=118 
n=80
n2=2
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

def g_func(x):
    g=1/(1+math.exp(-1*x))
    return g;

def logistic_reg_grad_descent(data,m,n,n2):
    X=data[0:n,0:n2]
    X=feat_scale(X,n,n2)
    X=np.concatenate((np.ones((n,1)),X),axis=1)
    Y=data[0:n,n2]
    theta=np.random.rand(n2+1,1)*10
    iteration=1000
    p=1.5
    theta1=theta
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
    X1=np.concatenate((np.ones((m-n,1)),X1),axis=1)
    Y1=data[n:,n2]
    tx=np.dot(X1,theta)
    O=np.random.rand(m-n,1)
    for a in range(m-n):
        if tx[a]>=0:
            O[a]=1
        else:
            O[a]=0
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

