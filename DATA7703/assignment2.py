# -*- coding: utf-8 -*-
"""
Created on Sun Sep  6 11:22:49 2020

@author: likeufo_ah
"""


import matplotlib.pyplot as plt
import numpy as np
from numpy import linalg as la
import operator
import sys

def load_data(filename):
    with open(filename, "r") as f:
        data=f.readlines()
        new_data=[]
        for i in data:
            rows=list(i.strip().split())
            new_data.append(rows)
    return new_data


def meanX(dataX):
    return np.mean(dataX,axis=0)

def pca(XMat, k):
    average = meanX(XMat) 
    m, n = np.shape(XMat)
    data_adjust = []
    avgs = np.tile(average, (m, 1))
    data_adjust = XMat - avgs
    covX = np.cov(data_adjust.T)
    featValue, featVec=  np.linalg.eig(covX)
    index = np.argsort(-featValue)
    finalData = []
    if k > n:
        print ("k must lower than feature number")
        return
    else:
        
        selectVec = np.matrix(featVec.T[index[:k]])
        finalData = data_adjust * selectVec.T 
        reconData = (finalData * selectVec) + average  
    return finalData, reconData,featValue,featVec,selectVec



def knn(traindata,testdata,labels,k=1):
    distances=np.linalg.norm(testdata-traindata,axis=1)
    sortDistance = distances.argsort()
    count = {}
    for i in range(k):
        vote = labels[sortDistance[i]]
        count[vote] = count.get(vote, 0) + 1
    sortCount = sorted(count.items(), key=operator.itemgetter(1), reverse=True)
    return sortCount[0][0]
 


def question1():
    data=load_data(file_train)
    dataX=np.array(data).reshape(280,1024)
    dataX=dataX.astype(np.float)
    data_mean=meanX(dataX)
    mean_picture=np.array(data_mean).reshape(32,32).T
    plt.imshow(mean_picture,cmap="gray")
    plt.show()
    

#eigenface=pca(dataX,5)[4]

def question1_2(k):
    data=load_data(file_train)
    dataX=np.array(data).reshape(280,1024)
    dataX=dataX.astype(np.float)
    p=pca(dataX, k)
    selectVec,recon=p[4],p[1]
    for i in selectVec:
        a=i.reshape(32,32).T
        a=a.astype(float)
        plt.imshow(a,cmap="gray")
       
        plt.show()
        

def question2(k):
    data=load_data(file_test)
    dataY=np.array(data)
    #print(dataY)
    dataY=dataY.astype(np.float)
    p=pca(dataY,k)
    reco=p[1]
    reco_pic=reco[2].reshape(32,32).T.astype(float)
    plt.imshow(reco_pic,cmap="gray")
    plt.title(f"k={k}")
    plt.show()

    





def question3_1nn(k):
    traindata=np.array(load_data(file_train))
    traindata=traindata.astype(np.float)
    #load trainset
    testdata=np.array(load_data(file_test))
    testdata=testdata.astype(np.float)
    #load testset
    with open(label_train, "r") as f:
        labels_train=f.read().split()
    #labels of trainset
    with open(label_test, "r") as f:
        labels_test=f.read().split() 
    #labels of testset
    pca_train=pca(traindata,k)[0]
    #train_dataset after dimension reduction
    
    average = meanX(testdata) 
    m, n = np.shape(testdata)
    avgs = np.tile(average, (m, 1))
    testdata = testdata - avgs
    pca_test= testdata * pca(traindata,k)[4].T
    #test dataset after dimension reduction
    '''
    X_values=[i for i in range(1,k)]
    Y_values=[question3_1nn(j) for j in X_values]
    plt.plot(X_values,Y_values)
    '''    
    
    sum=0.0
    wrong_value=[]
    right_value=[]
    index=[]
    for i,value in enumerate(pca_test):
        label=knn(pca_train,value,labels_train,k=1)
        
        label2=labels_test[i]
        
        if label==label2:
            sum=sum+1
        else:
            wrong_value.append(label)
            right_value.append(label2)
            index.append(i)
    accuracy=sum/len(labels_test)
    return accuracy,wrong_value,right_value,index

def question3_image(k):
    X_values=[i for i in range(1,k)]
    Y_values=[question3_1nn(j)[0]  for j in X_values]
    plt.plot(X_values,Y_values)
    
    
def question4(k):
    testdata=np.array(load_data(file_test))
    
    accuracy,wrong_value,right_value,index=question3_1nn(k)
    for i in index:
        pic=np.array(testdata[i])
        pic=pic.reshape(32,32).T
        pic=pic.astype(float)
        plt.imshow(pic,cmap="gray")
        plt.show()
    
    print("accuracy:",accuracy)
    return accuracy,wrong_value,right_value,index
    

    


if __name__ == "__main__":
    
    if len(sys.argv) < 6:
        
        print("not enough arguments specified")
        sys.exit(1)
    file_train=sys.argv[1]
    
    
    label_train=sys.argv[2]
    
    k=int(sys.argv[3])
    
    file_test=sys.argv[4]
    
    label_test=sys.argv[5]
    
    question1()
    question1_2(5)
    
    question2(5) 
    question2(10) 
    question2(50)
    question2(100)
    question2(k)# when input k,another image will be shown except for 4 images needed
            
    question3_1nn(100)
    question3_image(100)
    
    question4(100)
        
    

    
    
