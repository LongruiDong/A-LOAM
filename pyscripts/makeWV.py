#!/usr/bin/env python
# -*- coding: UTF8 -*-import numpy as np 
import os
import time
import numpy as np

import math


def fileToNumpy(filename):
    file = open(filename)
    file_lines = file.readlines()
    numberOfLines = len(file_lines)
    dataArray = np.zeros((numberOfLines, 4, 4))
    #labels = []
    index = 0
    for line in file_lines:
        line = line.strip() # 参数为空时，默认删除开头、结尾处空白符（包括'\n', '\r',  '\t',  ' ')
        formLine = line.split( ) #按空格切分？
        dataArray[index] = np.eye(4) 
        dataArray[index, 0, 0:4] = formLine[0:4]
        dataArray[index, 1, 0:4] = formLine[4:8]
        dataArray[index, 2, 0:4] = formLine[8:12]
        #labels.append((formLine[-1]))
        index += 1
    return dataArray

#
path_file = "/home/dlr/kitti/dataset/poses"
filename = "10.txt" #
print(filename)
fullpath = os.path.join(path_file, filename)
dataArray = fileToNumpy(fullpath)


T_CV = np.loadtxt("/home/dlr/kitti/dataset/sequences/10/Tr.txt")
Tcv = np.r_[T_CV,np.array([[0,0,0,1]])]

print(Tcv)
# Tvc = np.linalg.inv(Tcv) #注意这里！

for j in range(dataArray.shape[0]):
    # dataArray[j] = np.matmul(dataArray[j], Tvc) # Twv*Tvc=Twc
    dataArray[j] = np.matmul(dataArray[j], Tcv) #Twc * Tcv = Twv

T0 = dataArray[0] #(4,4)

#转为以初始帧为参考
T0_inv = np.linalg.inv(T0)
for k in range(dataArray.shape[0]): #
    T_k = dataArray[k]
    dataArray[k] = np.matmul(T0_inv, T_k)


f = open(path_file + "/10_lidar.txt", "w") #保存为新的gt位姿，初始为单位矩阵
for j in range(dataArray.shape[0]):
    slidt = [dataArray[j,0,0], dataArray[j,0,1], dataArray[j,0,2], dataArray[j,0,3],
                dataArray[j,1,0], dataArray[j,1,1], dataArray[j,1,2], dataArray[j,1,3],
                dataArray[j,2,0], dataArray[j,2,1], dataArray[j,2,2], dataArray[j,2,3]]
    k = 0
    for v in slidt:
        #if type(v) == 'torch.float32':
        #    v = v.numpy() 
        f.write(str(v))
        if(k<11): # 最后一个无空格，直接换行
            f.write(' ')
        k = k + 1
    f.write('\n')
f.close()


