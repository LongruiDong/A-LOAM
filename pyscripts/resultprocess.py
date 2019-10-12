#!/usr/bin/env python
# -*- coding: UTF8 -*-import numpy as np 
import os
import time
import numpy as np

import math

# import tf

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

path_file = "/home/dlr/catkin_ws2/src/A-LOAM/nresult2"
filename = "n04.txt" #
fullpath = os.path.join(path_file, filename)
dataArray = fileToNumpy(fullpath)
#m = dataArray.shape[0]
T0 = dataArray[0] #(4,4)
R0 = T0[0:3, 0:3]
R0_inv = np.linalg.inv(R0)

# T0_inv = np.linalg.inv(T0)
for k in range(dataArray.shape[0]): #1089
    T_k = dataArray[k]
    # R_k4 = np.eye(4)
    R_k = T_k[0:3, 0:3]
    # R_k4[0:3, 0:3] = R_k
    # quaternion = tf.transformations.quaternion_from_matrix(R_k4)
    # euler = tf.transformations.euler_from_quaternion(quaternion)
    # roll = -euler[1]
    # pitch = -euler[2]
    # yaw = euler[0]
    # tfmatrix = tf.transformations.compose_matrix(angles=np.array([roll, pitch, yaw]), 
    #                                             translate=np.array([T_k[0, 3], T_k[1, 3], T_k[2, 3]]))
    T_k[0:3, 0:3] = np.matmul(R0_inv, R_k)
    dataArray[k] = T_k
    # dataArray[k] = tfmatrix


f = open("/home/dlr/catkin_ws2/src/A-LOAM/nresult2/04.txt", "w") #保存为新的gt位姿，初始为单位矩阵
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


