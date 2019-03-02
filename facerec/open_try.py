# -*- coding: utf-8 -*-
"""
Created on Wed Jun 20 13:59:57 2018

@author: Jason
"""

import scipy.io as sio
from pylab import *
import numpy as np


matfn='YaleB_32x32.mat'
data=sio.loadmat(matfn)
face=data.get('fea')
gnd=data.get('gnd')
gnd.astype(np.int32)
    
n=np.zeros(39)
m=np.arange(39)
c=gnd.tolist()
for i in range(1,39):
    n[i-1]=c.index([i])

m=n.astype(int32)    
#关于这里的数据类型，好像是不可以随便转换的
m[38]=2413
test=[]
test_lab=[]
train=[]
train_lab=[] 
for i in range(0,38):
    test.extend(face[m[i]:m[i]+30,:])
    test_lab.extend(gnd[m[i]:m[i]+30])
    train.extend(face[m[i]+30:m[i+1],:])
    train_lab.extend(gnd[m[i]+30:m[i+1]])
   
#test=np.array(test)
#test_lab=np.array(test_lab)
#train=np.array(train)
#train_lab=np.array(train_lab)


#for i in range(1,39):
#    m[i-1]=c.index([i])