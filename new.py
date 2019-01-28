# -*- coding: utf-8 -*-
"""
Created on Thu Dec 27 09:37:06 2018

@author: Ishan
"""
import numpy as np
import matplotlib.pyplot as plt

def feat_scale(X):
    Xmax=np.max(X,axis=0)
    Xmin=np.min(X,axis=0)
    Xmean=np.mean(X,axis=0)
    X=np.divide((X-Xmean),(Xmax-Xmin))
    return X;

m=100
feature=2
training=70
data=np.loadtxt("ex2data1.txt" ,delimiter=',')
np.random.shuffle(data)
X1=feat_scale(data[:,0])
X2=feat_scale(data[:,1])
Y=data[:,feature] #100X1
plt.plot(X1[Y==1],X2[Y==1], 'r+',label='admitted')
plt.plot(X1[Y==0],X2[Y==0], 'g.',label='rejected')
Y=data[:training,feature]
Y=np.reshape(Y,(training,1))
X=data[:training,:feature]
X=feat_scale(X)
X=np.concatenate((np.ones((training,1)),X),axis=1) #100X3
theta=np.random.rand(1,feature+1) #1X3
#applying gradient descent
iterations=1500
alpha=0.01
for i in range(iterations):
    theta1=theta.copy()
    hypo=np.array(np.dot(X,theta.T)>0,dtype='int') #100X1
    theta1=theta-alpha*np.sum(((hypo-Y)*X),axis=0)/m
    theta=theta1.copy()
X0=feat_scale(data[training:,:feature])
X0=np.concatenate((np.ones((m-training,1)),X0),axis=1)
Y0=data[training:,feature]
hypo0=np.array(np.dot(X0,theta.T)>0,dtype='int')
x=np.linspace(-1,1,100)
plt.plot(x,(theta[0,0]+x*theta[0,1])*-1/theta[0,2] ,color='blue' ,label='decision boundary')
X01=X0[:,0]
X02=X0[:,1]
plt.plot(X01[Y0==1],X02[Y0==1],'b+')
plt.plot(X01[Y0==0],X02[Y0==0],'g.')
Y0=np.reshape(Y0,(m-training,1))
plt.xlabel('Marks in exam 1')
plt.ylabel("marks in exam 2")
plt.legend()
plt.title('Admission status')
plt.savefig('logistic_reg')
plt.show()
#accuracy
accuracy=np.array(hypo0==Y0, dtype='int')
accuracy=np.sum(accuracy)*100/(m-training)
print(accuracy)