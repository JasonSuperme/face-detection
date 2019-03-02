# -*- coding: utf-8 -*-
"""
Created on Wed Jun 20 17:53:18 2018

@author: Jason
"""

import numpy as np
import openfile as op
import matplotlib.pyplot as plt 

#假设输入的是测试集
#np.dot代表矩阵乘法  也就是点乘
#data 训练数据，label数据类型，k降到维数  lda最多降到c-1维 c为类别数
def lda(data,label,k):

    m,n=data.shape
    c=np.unique(label)#去掉标签中重复的项
    #对于np.mean  其中axis=0时求的时列的平均   axis=1时求的是行的平均
    data_mean=np.mean(data,0)#相当于求得所有的平均
    Sw=np.zeros((n,n),dtype=np.float64)
    Sb=np.zeros((n,n),dtype=np.float64)
    #这个循环时对单独的类进行操作
    #我们可以假使现在有38个类，然后我们对第一类进行操作
    for i in c:
        Xi=data[np.where(label==i)[0],:]#找到某一个类别,返回的是bool值 为真就输出对应的值，为假就不输出
        #Xi此时对应的就是所有的为第一类的图片
        meanClass=Xi.mean(axis=0)
        #假设目前有第一类共30张图片，那么就是这30张图片的平均值
        Sw=Sw+np.dot((Xi-meanClass).T,(Xi-meanClass))
        #得到的是一个1024*1024的矩阵，然后把这38类的1024矩阵加和
        Sb=Sb+n*np.dot((meanClass-data_mean).T,(meanClass-data_mean))
        #是将每一类的矩阵通过n*每一类单独的平均减去总体的平均，也是一个1024*1024的矩阵
    eigenVal,eigenVect=np.linalg.eig(np.linalg.inv(Sw)*Sb)
    #最终我们通过数学分解，得到的就是对Sw.T*Sb求特征向量的，这一步和pca基本上相同
    eigenVect=eigenVect[:,:k]#(1024,30)

    return eigenVect,data_mean

def findface(n):
    train_face,train_face_number,test_face,test_face_number = op.loadData(n)
    V ,mean= lda(train_face,train_face_number,30)
    train_face_new=np.dot(train_face,V)
    test_face_new=np.dot(test_face,V)

    num_train = train_face.shape[0]  #训练集个数
    num_test = test_face.shape[0]        #测试集个数

    true_num = 0  
    for i in range(num_test):
        testFace = test_face_new[i,:]
        diffMat = train_face_new- np.tile(testFace,(num_train,1))
        sqDiffMat = diffMat**2
        sqDistances = sqDiffMat.sum(axis=1)
        sortedDistIndicies = sqDistances.argsort()
        indexMin = sortedDistIndicies[0]
        if train_face_number[indexMin] == test_face_number[i]:
            true_num += 1
    accuracy = float(true_num)/num_test
    print('When use %d picture,The classify accuracy is: %.2f%%'%(n,accuracy * 100))
    return accuracy

def main():
   b=np.arange(25,35).tolist()
   a=[]
   for i in range(25,35):
        a.append(findface(i))
   plt.figure()
   plt.xlim(0,40)
   plt.ylim(0,1)
   plt.plot(b,a,'r')
    
    
if __name__=='__main__':
    main()
   
        
