# -*- coding: utf-8 -*-
"""
Created on Thu Dec 20 13:29:06 2018

@author: ISHAN
"""
import numpy as np
import math

def scaling(X):
    Xmax=np.max(X,axis=0)
    Xmin=np.min(X,axis=0)
    Xmean=np.mean(X,axis=0)
    X=np.divide(np.subtract(X,Xmean),np.subtract(Xmax,Xmin))
    return X

def hypo(theta,X):
    return np.dot(X,theta)

def sigmoid(X):
    m1=X.shape
    val=np.ones(m1)
    val=np.multiply(val,math.exp(1))
    X=np.multiply(X,-1)
    val=np.power(val,X)
    val=np.add(np.ones(m1),val)
    val=np.divide(np.ones(m1),val)
    return val

def costFunc(theta,X,Y):
    m1,n1=X.shape
    Y1=np.subtract(1,Y)
    H=sigmoid(hypo(theta,X))
    H1=np.subtract(1,H)
    H=np.log(H)
    H1=np.log(H1)
    cost=np.add(np.dot(Y.T,H),np.dot(Y1.T,H1))
    cost=np.divide(cost,-1*m1)
    return cost

def derCost(theta,X,Y):
    m1,n1=X.shape
    diff=np.subtract(sigmoid(hypo(theta,X)),Y)
    diff=np.dot(X.T,diff)
    return np.divide(diff,m1)

def gradientDescent(theta,X,Y,alpha):
    deri=derCost(theta,X,Y)
    theta=np.subtract(theta,np.multiply(deri,alpha))
    return theta

trainingExample=np.loadtxt(fname='ex2data1.txt' ,delimiter=',')
m,n=trainingExample.shape
n=n-1

X=np.copy(trainingExample[:,:n])
Y=trainingExample[:,n]
for i in range(n):
    x=np.ones((m,1))
    for j in range(m):
        x[j]=X[j,i]
    X=np.concatenate((X,np.multiply(x,x)),axis=1)
X=scaling(X)
n=2*n
ones=np.ones((m,1))
X=np.concatenate((ones,X,),axis=1)

theta=np.zeros(n+1)

alpha=0.1
iterations=1500

for i in range (iterations):
    theta=gradientDescent(theta,X,Y,alpha)
    
output=hypo(theta,X)
count=0
for i in range(m):
    if output[i]<0:
        output[i]=0
    else:
        output[i]=1
    if output[i]==Y[i]:
        count=count+1

print(count/m*100)