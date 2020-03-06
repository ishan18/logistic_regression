# -*- coding: utf-8 -*-
"""
Created on Sat Mar  7 00:32:06 2020

@author: RAJNIKANT
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

trainingExample=np.loadtxt(fname='ex2data2.txt' ,delimiter=',')
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
max1=np.max(Y)
min1=np.min(Y)
Y1=np.zeros((m,int(max1-min1+1)))
for i in range(m):
    Y1[i,int(Y[i]-min1)]=1

theta=np.zeros((n+1,int(max1-min1+1)))

alpha=0.1
iterations=1500

for j in range(int(max1-min1+1)):
    for i in range (iterations):
        theta[:,j]=gradientDescent(theta[:,j],X,Y1[:,j],alpha)

output=hypo(theta,X)

for i in range(m):
    max2=np.max(output[i])
    for j in range(int(max1-min1+1)):
        if output[i,j]==max2:
            output[i,0]=j+min1

#output[:,0] is the final predicted output

count=0
for i in range(m):
    if output[i,0]==Y[i]:
        count=count+1

print(count/m*100)