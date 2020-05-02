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

#/home/dlr/Downloads/imlsresult/imls-04.txt  /home/dlr/Downloads/imlsresult
path_file = "/home/dlr/imlslam/src/A-LOAM/result"
filename = "raw-04-calib-2imap-ds7orrdor.txt" #
fullpath = os.path.join(path_file, filename)
dataArray = fileToNumpy(fullpath)
print(filename[4:6])
# 使用数据集中给定的相机-lidar的校准文件把原始结果中的lidar坐标系转为相机坐标系！ calib_velo_to_cam.txt
# T_CV = np.load("/home/dlr/kitti/odometry_raw/2011_09_30/calib/T_CV.npy") #seq04-10

''' Tr: -1.857739385241e-03 -9.999659513510e-01 -8.039975204516e-03 -4.784029760483e-03 
     -6.481465826011e-03 8.051860151134e-03 -9.999466081774e-01 -7.337429464231e-02 
     9.999773098287e-01 -1.805528627661e-03 -6.496203536139e-03 -3.339968064433e-01 '''
 
# T_CV = np.load("/home/dlr/kitti/odometry_raw/2011_10_03/calib/T_CV.npy") #seq00-02
# T_CV = np.load("/home/dlr/kitti/odometry_raw/2011_09_26/calib/T_CV.npy") #seq 03
T_CV = np.loadtxt("/home/dlr/kitti/dataset/sequences/04/Tr.txt")
Tcv = np.r_[T_CV,np.array([[0,0,0,1]])]
# seq 04 calib.txt Tr 使用这个 平移上结果 更小？ 待验证
# Tcv = np.array([[-1.857739385241e-03,-9.999659513510e-01,-8.039975204516e-03,-4.784029760483e-03],
#                 [-6.481465826011e-03,8.051860151134e-03,-9.999466081774e-01,-7.337429464231e-02],
#                 [9.999773098287e-01,-1.805528627661e-03,-6.496203536139e-03,-3.339968064433e-01],
#                 [0.000000e+00,0.000000e+00,0.000000e+00,1.000000e+00]])
print(Tcv)
'''
R_cv = np.array([[-1.857739385241e-03,-9.999659513510e-01,-8.039975204516e-03],
                 [-6.481465826011e-03,8.051860151134e-03,-9.999466081774e-01],
                 [9.999773098287e-01,-1.805528627661e-03,-6.496203536139e-03]])
R_m = np.array([[0,-1,0],
                 [0,0,-1],
                 [1,0,0]])
R_delta = np.matmul(R_cv, np.linalg.inv(R_m))
ola_cv = tf.transformations.euler_from_matrix(R_cv, axes='szyx')
ola_m = tf.transformations.euler_from_matrix(R_m, axes='rzyx')
ola_delta = tf.transformations.euler_from_matrix(R_delta, axes='szyx')
print(ola_cv)
print(ola_m)
print(ola_delta)
'''
Tvc = np.linalg.inv(Tcv) #注意这里！

for j in range(dataArray.shape[0]):
    dataArray[j] = np.matmul(dataArray[j], Tvc) # Twv*Tvc=Twc
    # dataArray[j] = np.matmul(dataArray[j], Tcv) #Twc * Tcv = Twv
#m = dataArray.shape[0]
T0 = dataArray[0] #(4,4)
# R0 = T0[0:3, 0:3]
# R0_inv = np.linalg.inv(R0)
#转为以初始帧为参考
T0_inv = np.linalg.inv(T0)
for k in range(dataArray.shape[0]): #1089
    T_k = dataArray[k]
    dataArray[k] = np.matmul(T0_inv, T_k)


# f = open(path_file + "/04-calib-2imap-scan.txt", "w") #保存为新的gt位姿，初始为单位矩阵
f = open(path_file + "/" + filename[4:], "w")
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


