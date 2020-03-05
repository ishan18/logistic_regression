# -*- coding: utf-8 -*-
"""
Created on Thu Dec 20 13:29:06 2018

@author: ISHAN
"""
import numpy as np
import math
import matplotlib.pyplot as plt

data=np.loadtxt('LogisticRegressionData1.txt', delimiter=',') #importing dataset
np.random.shuffle(data)
m=100 #total no. of datasets
training_example=int(m*70/100) #no. of training examples used
feature=2 #no. of features

#making a function for feature scaling
def feat_scale(X,training_example,feature):
    Xmax=np.max(X,axis=0)
    Xmin=np.min(X,axis=0)
    Xmean=np.mean(X,axis=0)
    X=np.divide((X-Xmean),(Xmax-Xmin))
    return X;

#making the 'g' function
def g_func(x):
    g=1/(1+math.exp(-1*x))
    return g;

#making a function to carry out logistic regression problem and 
#giving accuracy of the program by entering the required details
def logistic_reg_grad_descent(data,dataset,training_example,feature):
    #data=np.random.shuffle(data)
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
    X1=feat_scale(X1,m-training_example,feature)
    X1=np.concatenate((np.ones((m-training_example,1)),X1),axis=1) #because X0=1
    Y1=data[training_example:,feature]
    thetaX=np.dot(X1,theta)
    output=np.random.rand(m-training_example,1)
    #Assuming that if 'g' function gives a value >=0.5 then output 
    #is 1 else output is 0
    for a in range(m-training_example):
        if thetaX[a]>=0:
            output[a]=1
        else:
            output[a]=0
    print('Output is:' ,output)
    plt.plot(X[:,1],-1*(np.dot(np.delete(X,1,axis=1),np.delete(theta,1)))/theta[1],'r.')
    x=np.linspace(-1,1)
    plt.plot(x,-1*(theta[0]+theta[2]*x)/theta[1])
    plt.plot(X1[:,1],-1*(np.dot(np.delete(X1,1,axis=1),np.delete(theta,1)))/theta[1],'b.')
    plt.legend(['training', 'decision boundary','test'])
    plt.savefig('my_plot.png')
    #To calculate accuracy
    accuracy=np.random.rand(m-training_example,1)
    for k in range(m-training_example):
        if Y1[k]==output[k]:
            accuracy[k]=1
        else:
            accuracy[k]=0
    accuracy=np.sum(accuracy)
    accuracy=accuracy*100/(m-training_example)
    return accuracy;

accuracy=logistic_reg_grad_descent(data,m,training_example,feature)
print('accuracy' ,accuracy)
