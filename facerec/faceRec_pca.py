# -*- coding: utf-8 -*-
"""
Created on Tue Jun 19 12:20:41 2018

@author: Jason
"""

import numpy as np
import openfile as op
import matplotlib.pyplot as plt 


#算法描述
#均以在37张人脸中每个人来拿选取10张 图片作为训练集 也就是有370个 数据集
#将 要作为训练集的数据放进来  比如训练数据有370组，那么（370*32*32）的训练集
def pca(data,k):
    data=np.float32(np.mat(data))#np.mat将输入作为矩阵
    row,col=data.shape
    data_mean=np.mean(data,0)#求均值,求每一列的均值  也就是这380张图片整体的均值
    Z=data-np.tile(data_mean,(row,1))#np.tile是将值延展开来,按照列展开,相当于在同一行复制很多遍370*1024
    D,V=np.linalg.eig(Z*Z.T)   #计算协方差矩阵  同时提取特征值和特征向量
    #特征值为（370，1） 特征向量（370，370)  
    #这里的数据集有370个，因此特征向量维数为370，这370是对整个数据集而言的
    #我们在特征值中选取前k个特征向量，最终的数据集合只有k维
    ##################################################
    ##理解错了
    #本来特征向量应该是一个1024*1024的矩阵，因为你要将这1024维的图片降到30维
    #而不是一开始理解的将370维图片降到30维  所以理解有错误
    #那为什么这里会得到370*370维的图片呢，是因为这个方法可以简化运算
    #也就是说1024*1024不太容易得到  就用了这里的方式来将它计算
    #################################################
    
    eigValInd = np.argsort(D)#这个排列是从小到大的
    eigValInd = eigValInd[:-(k+1):-1]#加上前面的：作用是让它逆序，从而得到前k项
    V1= V[:,eigValInd]
    #这里可以换成通过所占的比重来定义
    #V1=V[:,:k]   #提取前k个作为特征值  这个方法和前面三行的代码含义是相同的
    #得到的V1是一个370*30的矩阵，也就是取了数据的前30维
    #这里的370*30得到的就是作为测试集最终投影后的形式
    V1=Z.T*V1  #得到1024*30的矩阵，这是将训练集投影到了选取的特征向量上
    #本来协防擦汗矩阵是370*370  现在我用30维刻画了这370维  如果我用370维  那么数据不变
    ###这里就是我所说的理解错误的地方，其实是通过370*370来求1024*30的
    for i in range(k):
        V1[:,i]/=np.linalg.norm(V1[:,i])  #归一化（np.linalg.norm是个求范数的工具，相当于sqrt(x1**2+x2**2+...+xn**2)
                                            #整体除这个就相当于是归一化了

    return np.array(Z*V1),data_mean,V1

def facefind(n,k): 
    train_face,train_face_number,test_face,test_face_number = op.loadData(n)
    data_train_new,data_mean,V = pca(train_face,k)
    #其实这里得到的data_train_new是先归一化然后和原数据相乘的，我不明白这里为什么要归一化
    #同时V得到的也是归一化后的V，而且就是带着投影的那个值    
    num_train = data_train_new.shape[0]  #训练集个数
    num_test = test_face.shape[0]       #测试集个数
    temp_face = test_face - np.tile(data_mean,(num_test,1))#减去训练集的
    data_test_new = temp_face*V #得到测试脸在特征向量下的数据
    #这里应该得到的是1980*30
    data_test_new = np.array(data_test_new)#在一开始用np.mat将图像转化为了矩阵形式，这里是要将矩阵还原
    #为数组，为了能够进行类别区分
    data_train_new = np.array(data_train_new)
    true_num = 0
    for i in range(num_test):
        testFace = data_test_new[i,:]#得到的是某一张图1*50的一个数组
        diffMat = data_train_new - np.tile(testFace,(num_train,1))
        sqDiffMat = diffMat**2#通过欧式距离得到类别
        sqDistances = sqDiffMat.sum(axis=1)#将每一行加起来，取最大，每一行代表一个类别
        sortedDistIndicies = sqDistances.argsort()#对结果排序
        indexMin = sortedDistIndicies[0]#取出最小的，它的下标，也就是距离最近的
        if train_face_number[indexMin] == test_face_number[i]:
            true_num += 1#判断对的个数

    accuracy = float(true_num)/num_test
    print('When use %.2f%% picture of all picture as training set,The classify accuracy is: %.2f%%'%(num_train/2414*100,accuracy * 100))
    return accuracy


def main():
    
        b=np.arange(5,30).tolist()
    #k=[20,50,100,200]

    #for j in k:
        a=[]
        for i in range(5,30):
            a.append(facefind(i,150))   
        plt.figure()
        plt.xlim(0,40)
        plt.ylim(0,1)
        plt.plot(b,a,'r')
 
    
if __name__=='__main__':
    main()



    
    