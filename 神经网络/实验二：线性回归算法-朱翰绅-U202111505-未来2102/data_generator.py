import numpy as np
from typing import Optional,List,Tuple,Union
from numpy import ndarray
"""
产生两个都具有200个二维向量的数据集X_1和X_2。
数据集X_1的样本来自均值向量〖m_1=[-5,0]〗^T 、
协方差矩阵s_1=I的正态分布，属于“+1”类，
数据集X_2的样本来自均值向量〖m_2=[0,5]〗^T、
协方差矩阵s_2=I的正态分布，属于“-1”类，其中是一个2*2的单位矩阵。
产生的数据中80%用于训练，20%用于测试。

重复第2题的内容，但数据集X_1和数据集X_2的均值向量分别改为
〖m_1=[1,0]〗^T和〖m_2=[0,1]〗^T，其他不变。
"""
def pla_dataset(mean:List=None,matrix:List=[[1,0],[0,1]],size:int=None,
                class_label:int=None,split:bool=False,train_test_ratio:float=0.8,):
    np.random.seed(60)
    data=np.random.multivariate_normal(mean,matrix,size)
    label=np.ones(size).reshape(size,-1)
    if class_label is not None:
        label[:]=class_label
        data=np.concatenate((data,label),axis=1)
    if split:
        index=int(train_test_ratio*size)
        return data[:index,:],data[index:,:]
    else:
        return data

def homework_dataset():
    dataset=np.array( [[0.2,0.7,1.0],
                       [0.3,0.3,1.0],
                       [0.4,0.5,1.0],
                       [0.6,0.5,1.0],
                       [0.1,0.4,1.0],
                       [0.4,0.6,-1.0],
                       [0.6,0.2,-1.0],
                       [0.7,0.4,-1.0],
                       [0.8,0.6,-1.0],
                       [0.7,0.5,-1.0]])
    return dataset
