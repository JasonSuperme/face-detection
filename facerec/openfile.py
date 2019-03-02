# -*- coding: utf-8 -*-
"""
Created on Tue Jun 19 11:47:42 2018

@author: Jason
"""

import scipy.io as sio
import numpy as np

def loadData(num):

    matfn='YaleB_32x32.mat'
    data=sio.loadmat(matfn)
    face=data.get('fea')
    gnd=data.get('gnd')
    
    n=np.zeros(39)
    m=np.arange(39)
    c=gnd.tolist()
    for i in range(1,39):
        n[i-1]=c.index([i])

    m=n.astype(np.int32)
#关于这里的数据类型，好像是不可以随便转换的

    m[38]=2413
    test=[]
    test_lab=[]
    train=[]
    train_lab=[] 
    for i in range(0,37):
        train.extend(face[m[i]:m[i]+num,:])
        train_lab.extend(gnd[m[i]:m[i]+num])
        test.extend(face[m[i]+num:m[i+1],:])
        test_lab.extend(gnd[m[i]+num:m[i+1]])
        
    test=np.array(test)
    test_lab=np.array(test_lab)
    train=np.array(train)
    train_lab=np.array(train_lab)
    return train,train_lab,test,test_lab

