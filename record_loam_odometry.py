#!/usr/bin/env python
# -*- coding: UTF8 -*-
# Autor> Clayder Gonzalez

import time
import rospy
import math
import numpy as np
import tf

from nav_msgs.msg import Odometry     
from geometry_msgs.msg import Twist
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import String

import matplotlib
import matplotlib.pyplot as plt
import math

trajectoryFile = None
timeStamp = None
# pitch = None

def odom_callback_loam(data):
	# global trajectoryFile
	global newtraj
	global timeStamp
	global tfmatrix
	# global tfm
	# global Tcv
	# 使用数据集中给定的相机-lidar的校准文件把原始结果中的lidar坐标系转为相机坐标系！ calib_velo_to_cam.txt
	# T_CV = np.load("/home/dlr/kitti/odometry_raw/2011_09_30/calib/T_CV.npy") #(3,4) seq04-10
	# T_CV = np.load("/home/dlr/kitti/odometry_raw/2011_10_03/calib/T_CV.npy") #seq00-02
	# T_CV = np.load("/home/dlr/kitti/odometry_raw/2011_09_26/calib/T_CV.npy") #seq 03
	# Tcv = np.r_[T_CV,np.array([[0,0,0,1]])]
	# Tvc = np.linalg.inv(Tcv) #那不应该取逆啊  试试取逆的结果
	timeStampNum = data.header.stamp.secs + (data.header.stamp.nsecs * 10**(-9))
	timeStamp = str(timeStampNum)
	quaternion = (data.pose.pose.orientation.x, data.pose.pose.orientation.y, data.pose.pose.orientation.z, data.pose.pose.orientation.w)
	# 虽然四元数这里不影响旋转的角度误差的计算，为了分析分量上的误差(保证其对应)，矫正一下 不用在进行result process
	# quaternion = (-data.pose.pose.orientation.y, -data.pose.pose.orientation.z, data.pose.pose.orientation.x, data.pose.pose.orientation.w)
	euler = tf.transformations.euler_from_quaternion(quaternion)
	# roll = -euler[1]
	# pitch = -euler[2]
	# yaw = euler[0]
	# 调整了原始的“世界坐标系” 视图和kitti benchmark对齐 注意这里输入的 (euler)
	# tfmatrix = tf.transformations.compose_matrix(angles=np.array([roll, pitch, yaw]), translate=np.array([data.pose.pose.position.x, 
	# 														data.pose.pose.position.y, data.pose.pose.position.z]))
	tfmatrix = tf.transformations.compose_matrix(angles=np.array(euler), translate=np.array([data.pose.pose.position.x, 
															data.pose.pose.position.y, data.pose.pose.position.z]))
	
	# 转换后的pose
	# tfm = np.matmul(Tvc, tfmatrix)
	# tfm = np.matmul(tfmatrix, Tvc)
	# trajectoryFile.write(timeStamp + " " + str(data.pose.pose.position.z) + " " + str(data.pose.pose.position.x) + " " + str(pitch) + "\n")
	newtraj.write(str(tfmatrix[0, 0]) + " " + str(tfmatrix[0, 1]) + " " + str(tfmatrix[0, 2]) + " " + str(tfmatrix[0, 3])
						 + " " + str(tfmatrix[1, 0]) + " " + str(tfmatrix[1, 1]) + " " + str(tfmatrix[1, 2]) + " " + str(tfmatrix[1, 3])
						 + " " + str(tfmatrix[2, 0]) + " " + str(tfmatrix[2, 1]) + " " + str(tfmatrix[2, 2]) + " " + str(tfmatrix[2, 3]) + "\n")
	print(timeStamp)	

def writeFile(file):
	# global trajectoryFile
	global newtraj
	# inicializa nodo ROS
	rospy.init_node('record_loam_odometry')
	# trajectoryFile = open(file, 'w')
	newtraj = open(file, 'w') # write standard traj format
	# rospy.Subscriber('/integrated_to_init', Odometry, odom_callback_loam)
	rospy.Subscriber('/aft_mapped_to_init', Odometry, odom_callback_loam)
	rospy.spin()
	# trajectoryFile.close()
	newtraj.close()
        
if __name__ == '__main__':
	writeFile('/home/dlr/myimls/src/A-LOAM/result/04raw665-l015.txt')
