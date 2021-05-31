import pandas as pd
import math
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics

train_data = np.zeros((10000,50))
test_data = np.zeros((10000,50))

def open_data():
    train_data = np.load('./data/train_data.npy')
    test_data  = np.load('./data/test_data.npy')
    
    return train_data,test_data

#######################################################
#因為train data前5000筆為class 1的前5000筆;後5000筆為class 2的前5000筆
#test data也是
#所以class 1 Label為0，class 2 Label為1
#######################################################    
def label_data():
    class_1 = [0]*5000
    class_2 = [1]*5000
    
    y_train =  np.concatenate((class_1,class_2),axis=0)
    y_test = np.concatenate((class_1,class_2),axis=0)
    print(y_train)
    return y_train,y_test
    
def Bayes_Classifier():
     gnb = GaussianNB()
     gnb.fit(x_train, y_train)
     y_pred = gnb.predict(x_test)
     print("Accuracy:",metrics.accuracy_score(y_test, y_pred))




if __name__ =="__main__":
    
    #open data
    x_train,x_test = open_data()
    
    #label
    y_train,y_test = label_data()
    
    #Bayes Classifier
    Bayes_Classifier()
    
 
    
    