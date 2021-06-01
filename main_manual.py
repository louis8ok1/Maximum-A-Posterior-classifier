import pandas as pd
import math
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics

train_data = np.zeros((10000,50))
test_data = np.zeros((10000,50))

new_train_data = np.zeros((10000,50))
new_test_data = np.zeros((10000,50))
def open_data():
    train_data = np.load('./data/train_data.npy')
    test_data  = np.load('./data/test_data.npy')
    #print(train_data)
    return train_data,test_data

def label_data():
    class_1 = [[0]]*5000
    class_2 = [[1]]*5000
    
    y_train =  np.concatenate((class_1,class_2),axis=0)
    y_test = np.concatenate((class_1,class_2),axis=0)
    #print(y_train)
    return y_train,y_test

def concatenate_Data(x_train,x_test,y_train,y_test):
    new_train_data = np.concatenate((x_train,y_train),axis = 1)
    new_test_data  = np.concatenate((x_test,y_test),axis = 1)
   # print(new_train_data)
    print(new_train_data.shape)
    return new_train_data,new_test_data

def mean(cls1_test_data,cls2_test_data):
    temp1  = np.array([[]])
    temp2  = np.array([[]])
    temp1 = np.mean(cls1_test_data, axis=0)
    temp1 = temp1[:50]
    temp2 = np.mean(cls2_test_data, axis=0)
    temp2 = temp2[:50]
    return temp1 , temp2

def cov(cls1_test_data,cls2_test_data):
    
    return np.cov(cls1_test_data[:,:50].T),np.cov(cls2_test_data[:,:50].T)

def test_data_split(new_test_data):
    #split data
    cls1_data = np.array([[]])
    cls2_data = np.array([[]])
    
    cls1_data = new_test_data[:5000,:]
    cls2_data =new_test_data[5000:10000,:]
    return cls1_data,cls2_data

def PDF(x,mu,cov):
     diff_x_mu = np.array([[]])

     for i in range(50):
         if i == 0 :
             diff_x_mu = np.append(diff_x_mu,[[ float(x[i][0])-mu[i] ]],axis=1)
         else :
             diff_x_mu = np.append(diff_x_mu,[[ float(x[i][0])-mu[i] ]],axis=0)
     #print(diff_x_mu.shape)       
     covinv = np.linalg.inv(cov)
     temp = np.dot(diff_x_mu.T,covinv)
     temp = np.dot(temp,diff_x_mu)
     tempexp = math.exp(temp*(-1/2))
     tempdet = np.linalg.det(cov)
     tempdet = math.pow(tempdet,0.5)
     demo = math.pow((math.pi)*2,25)
     demo *= tempdet
     p = tempexp/demo
     return p
def test_data_predict(cls1_test_data,cls2_test_data,mean1,mean2,cov1,cov2):
    cls1_test_data = cls1_test_data[:,:50]
    cls2_test_data = cls2_test_data[:,:50]
    print(cls1_test_data)
    cls1_test_data = cls1_test_data.T
    cls2_test_data = cls2_test_data.T
    print(cls1_test_data.shape)
    #test data, label is class 1 
    correct1 = 0 
    for j in range(5000):
        x = np.array([[]])
        for i in range(50):
            if i==0 :
                x = np.append(x,[[cls1_test_data[i,j]]],axis = 1)
            else :
                x = np.append(x,[[cls1_test_data[i,j]]],axis = 0)  
 
        if j == 0 :
            print(x)
        P_x_1 = PDF(x,mean1,cov1)    
        P_x_2 = PDF(x,mean2,cov2)
        if P_x_1 >=P_x_2 :
            correct1 +=1
    
    #test data, label is class 2
    correct2 = 0 
    for j in range(5000):
        x = np.array([[]])
        for i in range(50):
            if i==0 :
                x = np.append(x,[[cls2_test_data[i,j]]],axis = 1)
            else :
                x = np.append(x,[[cls2_test_data[i,j]]],axis = 0)  
 
            
        P_x_1 = PDF(x,mean1,cov1)    
        P_x_2 = PDF(x,mean2,cov2)
        if P_x_1 <=P_x_2 :
            correct2 +=1

    
    print("class1 test correct : ",correct1)
    print("class2 test correct : ",correct2)
    print("Accuracy of prediction : ", round(((correct1+correct2)/10000)*100,2),"%" )
    
if __name__ =="__main__":
    
    #open data
    x_train,x_test = open_data()
    #label
    y_train,y_test = label_data()
      
    #concatenate data & label
    new_train_data,new_test_data = concatenate_Data(x_train,x_test,y_train,y_test)
    print(new_test_data.shape)
    
    
     
    cls1_test_data,cls2_test_data = test_data_split(new_test_data)
    
    #mean vector 50*1
    mean1,mean2 = mean(cls1_test_data,cls2_test_data)
    
    #covariance matrix 50*50
    cov1,cov2 = cov(cls1_test_data,cls2_test_data)
    
    #test data predict
    test_data_predict(cls1_test_data,cls2_test_data,mean1,mean2,cov1,cov2)
    