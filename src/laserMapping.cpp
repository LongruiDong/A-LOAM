// This is an advanced implementation of the algorithm described in the following paper:
//   J. Zhang and S. Singh. LOAM: Lidar Odometry and Mapping in Real-time.
//     Robotics: Science and Systems Conference (RSS). Berkeley, CA, July 2014. 

// Modifier: Tong Qin               qintonguav@gmail.com
// 	         Shaozu Cao 		    saozu.cao@connect.ust.hk


// Copyright 2013, Ji Zhang, Carnegie Mellon University
// Further contributions copyright (c) 2016, Southwest Research Institute
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
// 3. Neither the name of the copyright holder nor the names of its
//    contributors may be used to endorse or promote products derived from this
//    software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

#include <math.h>
#include <vector>
#include <numeric>//用于给 vector 求和
#include <aloam_velodyne/common.h>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <geometry_msgs/PoseStamped.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <ros/ros.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/PointCloud2.h>
#include <tf/transform_datatypes.h>
#include <tf/transform_broadcaster.h>
#include <eigen3/Eigen/Dense>
#include <ceres/ceres.h>
#include <mutex>
#include <queue>
#include <thread>
#include <iostream>
#include <string>

#include "lidarFactor.hpp"
#include "aloam_velodyne/common.h"
#include "aloam_velodyne/tic_toc.h"

#include <typeinfo>
#include <iostream>
#include <pcl/common/common.h>
#include <pcl/ModelCoefficients.h>//模型系数
#include <pcl/io/pcd_io.h>
#include <pcl/sample_consensus/method_types.h>//采样方法
#include <pcl/sample_consensus/model_types.h>//采样模型
#include <pcl/segmentation/sac_segmentation.h>//随机采样分割
#include <pcl/filters/extract_indices.h>//根据索引提取点云
#include <pcl/features/normal_3d.h>//点云法线特征
#include <pcl/features/normal_3d_omp.h>
#include <pcl/kdtree/kdtree.h>//kd树搜索算法
#include <pcl/segmentation/extract_clusters.h>//欧式聚类分割
#include <pcl/features/feature.h>
#include <pcl/common/centroid.h>
#include <cmath>
using namespace std;

int frameCount = 0;

//时间戳
double timecloudprocessed = 0;
double timeLaserCloudCornerLast = 0;
double timeLaserCloudSurfLast = 0;
double timeLaserCloudFullRes = 0;
double timeLaserOdometry = 0;

//keep n =100 scan in the map
#define mapsize 100
//从每个scan list中采样的点数s=100
#define numselect 100
//采样和匹配中的 近邻距离阈值
#define RadThr 0.20
//定义IMLS surface的参数 h
#define h_imls 0.06

// input: from laserodometry.cpp  接收到的边沿点和平面点
pcl::PointCloud<PointType>::Ptr laserCloudCornerLast(new pcl::PointCloud<PointType>());
pcl::PointCloud<PointType>::Ptr laserCloudSurfLast(new pcl::PointCloud<PointType>());

//当前帧采样之后用于maping的点  9*100=900
pcl::PointCloud<PointType>::Ptr CloudSampled(new pcl::PointCloud<PointType>());


//input & output: points in one frame. local --> global 一帧的所有点
pcl::PointCloud<PointType>::Ptr laserCloudFullRes(new pcl::PointCloud<PointType>());
pcl::PointCloud<PointType>::Ptr Cloudprocessed(new pcl::PointCloud<PointType>());
// ouput: all visualble cube points 立方体点？ 匹配使用的特征点（下采样之后）
pcl::PointCloud<PointType>::Ptr laserCloudSurround(new pcl::PointCloud<PointType>());


//存放n=100个已经注册的之前的点云作为map
// pcl::PointCloud<PointType>::Ptr ModelPointCloud[mapsize];
std::vector<pcl::PointCloud<PointType>> ModelPointCloud(mapsize);
//从当前地图指针数组中提取的点
pcl::PointCloud<PointType>::Ptr laserCloudFromMap(new pcl::PointCloud<PointType>());

//kd-tree
pcl::KdTreeFLANN<PointType>::Ptr kdtreeFromMap(new pcl::KdTreeFLANN<PointType>());
pcl::KdTreeFLANN<PointType>::Ptr kdtreeProcessed(new pcl::KdTreeFLANN<PointType>());//用于计算当前scan中每个点的法线

//？  用于优化的变换矩阵
double parameters[7] = {0, 0, 0, 1, 0, 0, 0};
Eigen::Map<Eigen::Quaterniond> q_w_curr(parameters); //？ 前四个参数   
Eigen::Map<Eigen::Vector3d> t_w_curr(parameters + 4);
// std::cout<<"initial q_w_curr = \n"<<q_w_curr.coeffs() << endl;//debug x,y,z，w 是0 0 0 1 单位阵
// std::cout<<"initial t_w_curr = \n"<<t_w_curr.coeffs() << endl;
// wmap_T_odom * odom_T_curr = wmap_T_curr;
// transformation between odom's world and map's world frame
Eigen::Quaterniond q_wmap_wodom(1, 0, 0, 0); // w,x,y,z 地图的世界坐标和里程计的世界坐标两者是一致的
Eigen::Vector3d t_wmap_wodom(0, 0, 0);

//当前帧相对于 odom world的变换
Eigen::Quaterniond q_wodom_curr(1, 0, 0, 0);
Eigen::Vector3d t_wodom_curr(0, 0, 0);

//缓存来自laserodometry.cpp的量
std::queue<sensor_msgs::PointCloud2ConstPtr> processedBuf; //缓存cloud_4的
std::queue<sensor_msgs::PointCloud2ConstPtr> cornerLastBuf;
std::queue<sensor_msgs::PointCloud2ConstPtr> surfLastBuf;
std::queue<sensor_msgs::PointCloud2ConstPtr> fullResBuf;
std::queue<nav_msgs::Odometry::ConstPtr> odometryBuf;
std::mutex mBuf;

//创建voxelgrid滤波器 （体素栅格滤波器）
// pcl::VoxelGrid<PointType> downSizeFilterCorner;
// pcl::VoxelGrid<PointType> downSizeFilterSurf;

std::vector<int> pointSearchInd;
std::vector<float> pointSearchSqDis;

PointType pointOri, pointSel;//原始点，变换后的点

ros::Publisher pubLaserCloudSurround, pubLaserCloudMap, pubLaserCloudFullRes, pubOdomAftMapped, pubLaserAfterMappedPath, pubCloudSampled, pubCloudProcessed;

nav_msgs::Path laserAfterMappedPath;

//9种特征数组 保存每个点的特征值 用于比较大小
float samplefeature1[150000];
float samplefeature2[150000];
float samplefeature3[150000];
float samplefeature4[150000];
float samplefeature5[150000];
float samplefeature6[150000];
float samplefeature7[150000];
float samplefeature8[150000];
float samplefeature9[150000];
//序号
int cloudSortInd1[150000];
int cloudSortInd2[150000];
int cloudSortInd3[150000];
int cloudSortInd4[150000];
int cloudSortInd5[150000];
int cloudSortInd6[150000];
int cloudSortInd7[150000];
int cloudSortInd8[150000];
int cloudSortInd9[150000];
//比较大小
//比较两点曲率 从大到小降序排列
bool comp1 (int i,int j) { return (samplefeature1[i]>samplefeature1[j]); }
bool comp2 (int i,int j) { return (samplefeature2[i]>samplefeature2[j]); }
bool comp3 (int i,int j) { return (samplefeature3[i]>samplefeature3[j]); }
bool comp4 (int i,int j) { return (samplefeature4[i]>samplefeature4[j]); }
bool comp5 (int i,int j) { return (samplefeature5[i]>samplefeature5[j]); }
bool comp6 (int i,int j) { return (samplefeature6[i]>samplefeature6[j]); }
bool comp7 (int i,int j) { return (samplefeature7[i]>samplefeature7[j]); }
bool comp8 (int i,int j) { return (samplefeature8[i]>samplefeature8[j]); }
bool comp9 (int i,int j) { return (samplefeature9[i]>samplefeature9[j]); }			



// set initial guess 
//当前帧相对于map world 的变换
void transformAssociateToMap()
{
	q_w_curr = q_wmap_wodom * q_wodom_curr;
	t_w_curr = q_wmap_wodom * t_wodom_curr + t_wmap_wodom;
}

//
void transformUpdate()
{// transformAssociateToMap() 的逆过程？
	q_wmap_wodom = q_w_curr * q_wodom_curr.inverse();
	t_wmap_wodom = t_w_curr - q_wmap_wodom * t_wodom_curr;
}

//将（新一帧）点注册到map 世界坐标系下
void pointAssociateToMap(PointType const *const pi, PointType *const po)
{
	Eigen::Vector3d point_curr(pi->x, pi->y, pi->z);
	Eigen::Vector3d point_w = q_w_curr * point_curr + t_w_curr;
	po->x = point_w.x();
	po->y = point_w.y();
	po->z = point_w.z();
	po->intensity = pi->intensity;
	//po->intensity = 1.0;
}

/*
//转移到局部坐标系 是pointAssociateToMap（）的逆过程？  not used
void pointAssociateTobeMapped(PointType const *const pi, PointType *const po)
{
	Eigen::Vector3d point_w(pi->x, pi->y, pi->z);
	Eigen::Vector3d point_curr = q_w_curr.inverse() * (point_w - t_w_curr);
	po->x = point_curr.x();
	po->y = point_curr.y();
	po->z = point_curr.z();
	po->intensity = pi->intensity;
}
*/
//接收laserodo.cpp传来的三类点
void laserCloudCornerLastHandler(const sensor_msgs::PointCloud2ConstPtr &laserCloudCornerLast2)
{
	mBuf.lock();
	cornerLastBuf.push(laserCloudCornerLast2);
	mBuf.unlock();
}

void laserCloudSurfLastHandler(const sensor_msgs::PointCloud2ConstPtr &laserCloudSurfLast2)
{
	mBuf.lock();
	surfLastBuf.push(laserCloudSurfLast2);
	mBuf.unlock();
}

void laserCloudFullResHandler(const sensor_msgs::PointCloud2ConstPtr &laserCloudFullRes2)
{
	mBuf.lock();
	fullResBuf.push(laserCloudFullRes2);
	mBuf.unlock();
}


void CloudprocessedHandler(const sensor_msgs::PointCloud2ConstPtr &CloudProcessed2)
{
	mBuf.lock();
	processedBuf.push(CloudProcessed2);
	mBuf.unlock();
}

//receive odomtry  高频发布 相对于地图 world 坐标系的轨迹 /aft_mapped
void laserOdometryHandler(const nav_msgs::Odometry::ConstPtr &laserOdometry)
{
	mBuf.lock();
	odometryBuf.push(laserOdometry);
	mBuf.unlock();

	// high frequence publish
	Eigen::Quaterniond q_wodom_curr;
	Eigen::Vector3d t_wodom_curr;
	q_wodom_curr.x() = laserOdometry->pose.pose.orientation.x;
	q_wodom_curr.y() = laserOdometry->pose.pose.orientation.y;
	q_wodom_curr.z() = laserOdometry->pose.pose.orientation.z;
	q_wodom_curr.w() = laserOdometry->pose.pose.orientation.w;
	t_wodom_curr.x() = laserOdometry->pose.pose.position.x;
	t_wodom_curr.y() = laserOdometry->pose.pose.position.y;
	t_wodom_curr.z() = laserOdometry->pose.pose.position.z;

	/*
	//装换为map world frame为参考
	Eigen::Quaterniond q_w_curr = q_wmap_wodom * q_wodom_curr;
	Eigen::Vector3d t_w_curr = q_wmap_wodom * t_wodom_curr + t_wmap_wodom; 

	
	nav_msgs::Odometry odomAftMapped;
	odomAftMapped.header.frame_id = "/camera_init";
	odomAftMapped.child_frame_id = "/aft_mapped";
	odomAftMapped.header.stamp = laserOdometry->header.stamp;
	odomAftMapped.pose.pose.orientation.x = q_w_curr.x();
	odomAftMapped.pose.pose.orientation.y = q_w_curr.y();
	odomAftMapped.pose.pose.orientation.z = q_w_curr.z();
	odomAftMapped.pose.pose.orientation.w = q_w_curr.w();
	odomAftMapped.pose.pose.position.x = t_w_curr.x();
	odomAftMapped.pose.pose.position.y = t_w_curr.y();
	odomAftMapped.pose.pose.position.z = t_w_curr.z();
	pubOdomAftMappedHighFrec.publish(odomAftMapped);
	*/
}

//IMLS中的采样以及mapping过程（借鉴下面的void process() ）
void imls()
{
	while(1)
	{
		while (!cornerLastBuf.empty() && !surfLastBuf.empty() && //确保接收到laserodo.cpp发出的点云
			!fullResBuf.empty()  && !processedBuf.empty() && !odometryBuf.empty())
		{
			mBuf.lock();
			//确保各容器数据的时间戳是合理的
			while (!odometryBuf.empty() && odometryBuf.front()->header.stamp.toSec() < cornerLastBuf.front()->header.stamp.toSec())
				odometryBuf.pop();
			if (odometryBuf.empty())
			{
				mBuf.unlock();
				break;
			}

			while (!surfLastBuf.empty() && surfLastBuf.front()->header.stamp.toSec() < cornerLastBuf.front()->header.stamp.toSec())
				surfLastBuf.pop();
			if (surfLastBuf.empty())
			{
				mBuf.unlock();
				break;
			}

			while (!fullResBuf.empty() && fullResBuf.front()->header.stamp.toSec() < cornerLastBuf.front()->header.stamp.toSec())
				fullResBuf.pop();
			if (fullResBuf.empty())
			{
				mBuf.unlock();
				break;
			}

			//待采样的cloud
			while (!processedBuf.empty() && processedBuf.front()->header.stamp.toSec() < cornerLastBuf.front()->header.stamp.toSec())
				processedBuf.pop();
			if (processedBuf.empty())
			{
				mBuf.unlock();
				break;//无数据，重新接收，并执行判断是否接收到
			}

			//得到时间戳
			timecloudprocessed = processedBuf.front()->header.stamp.toSec();
			timeLaserCloudCornerLast = cornerLastBuf.front()->header.stamp.toSec();
			timeLaserCloudSurfLast = surfLastBuf.front()->header.stamp.toSec();
			timeLaserCloudFullRes = fullResBuf.front()->header.stamp.toSec();
			timeLaserOdometry = odometryBuf.front()->header.stamp.toSec();

			//确保各类点云的时间戳同步（那前面为啥也判断了时间？）
			if (timeLaserCloudCornerLast != timeLaserOdometry ||
				timeLaserCloudSurfLast != timeLaserOdometry ||
				timeLaserCloudFullRes != timeLaserOdometry || 
				timecloudprocessed != timeLaserOdometry)
			{
				printf("time corner %f surf %f full %f odom %f processed %f \n", timeLaserCloudCornerLast, timeLaserCloudSurfLast, timeLaserCloudFullRes, timeLaserOdometry, timecloudprocessed);
				printf("unsync messeage!");
				mBuf.unlock();
				break;
			}

			//数据从容器到点云指针
			laserCloudCornerLast->clear();
			pcl::fromROSMsg(*cornerLastBuf.front(), *laserCloudCornerLast);
			cornerLastBuf.pop();

			laserCloudSurfLast->clear();
			pcl::fromROSMsg(*surfLastBuf.front(), *laserCloudSurfLast);
			surfLastBuf.pop();

			laserCloudFullRes->clear();
			pcl::fromROSMsg(*fullResBuf.front(), *laserCloudFullRes);
			fullResBuf.pop();

			Cloudprocessed->clear();
			pcl::fromROSMsg(*processedBuf.front(), *Cloudprocessed);
			processedBuf.pop();

			//当前帧相对于odom world 的位姿  第0帧就是单位矩阵
			q_wodom_curr.x() = odometryBuf.front()->pose.pose.orientation.x;
			q_wodom_curr.y() = odometryBuf.front()->pose.pose.orientation.y;
			q_wodom_curr.z() = odometryBuf.front()->pose.pose.orientation.z;
			q_wodom_curr.w() = odometryBuf.front()->pose.pose.orientation.w;
			t_wodom_curr.x() = odometryBuf.front()->pose.pose.position.x;
			t_wodom_curr.y() = odometryBuf.front()->pose.pose.position.y;
			t_wodom_curr.z() = odometryBuf.front()->pose.pose.position.z;
			odometryBuf.pop();

			// while(!cornerLastBuf.empty())
			// {
			// 	cornerLastBuf.pop();//清空该容器和实时性有什么关系呢  是否因为这个才会有跳帧？ 注释后 不在跳帧了！
			// 	printf("drop lidar frame in mapping for real time performance \n");//那为啥不对其他容器再次pop一下呢
			// }

			mBuf.unlock();

			TicToc t_whole;

			//把odom的位姿转为相对于map世界坐标系的位姿 （实际上乘的是单位矩阵）
			transformAssociateToMap();//第一帧的话 应该有q_w_curr=1 0 0 0

			//********IMLS-SLAM SCAN SAMPLING STRATEGY 扫描点的采样
			TicToc t_scansample;//匹配前点云采样计时
			//先计算每个点的法线pcl::PointXYZI PointType
            pcl::NormalEstimationOMP<pcl::PointXYZI, pcl::Normal> ne;
            ne.setInputCloud (Cloudprocessed);

            pcl::search::KdTree<pcl::PointXYZI>::Ptr netree (new pcl::search::KdTree<pcl::PointXYZI>());
            ne.setSearchMethod (netree);

            pcl::PointCloud<pcl::Normal>::Ptr cloud_normals (new pcl::PointCloud<pcl::Normal>);
            //设置 半径内搜索临近点
            ne.setRadiusSearch (0.5);
            ne.compute (*cloud_normals);//得到当前scan：Cloudprocessed中每个点的法线

			//但是还要得到法线计算过程中的中间值 PCA的特征值
			kdtreeProcessed->setInputCloud(Cloudprocessed);
			int numprocessed = Cloudprocessed->points.size();
			double a_2d[numprocessed];//保存每个点的planar scalar
			
			//遍历每点计算a2d以及9个特征
			for (int i=0; i<numprocessed; i++)
			{
				pointSel = Cloudprocessed->points[i];
				//对于当前点在scan中搜索指定半径内的近邻
				kdtreeProcessed->radiusSearch(pointSel, 0.5, pointSearchInd, pointSearchSqDis);
				int numneighbor = pointSearchInd.size();//得到的半径内近邻个数
				std::vector<Eigen::Vector3d> neighbors;//存储若干近邻点
				Eigen::Vector3d center(0, 0, 0);//初始化近邻点的重心
				for (int j = 0; j < numneighbor; j++)
				{
					Eigen::Vector3d tmp(Cloudprocessed->points[pointSearchInd[j]].x,
										Cloudprocessed->points[pointSearchInd[j]].y,
										Cloudprocessed->points[pointSearchInd[j]].z);
					center = center + tmp;
					neighbors.push_back(tmp);
				}
				//得到近邻点坐标的重心
				center = center / double(numneighbor);

				Eigen::Matrix3d covMat = Eigen::Matrix3d::Zero();//近邻点的协方差矩阵
				for (int j = 0; j < numneighbor; j++)
				{
					Eigen::Matrix<double, 3, 1> tmpZeroMean = neighbors[j] - center;
					covMat = covMat + tmpZeroMean * tmpZeroMean.transpose();
				}

				Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> saes(covMat);//协方差矩阵特征值分解
				// note Eigen library sort eigenvalues in increasing order
				//得到三个特征值 从大到小
				double lamda_1 = sqrt(saes.eigenvalues()[2]);
				double lamda_2 = sqrt(saes.eigenvalues()[1]);
				double lamda_3 = sqrt(saes.eigenvalues()[0]);
				//计算当前点近邻的planar scale
				a_2d[i] = (lamda_2 - lamda_3)/lamda_1;

				Eigen::Vector3d X_axis(1, 0, 0);
				Eigen::Vector3d Y_axis(0, 1, 0);
				Eigen::Vector3d Z_axis(0, 0, 1);

				Eigen::Vector3d pointcurr(Cloudprocessed->points[i].x,
										  Cloudprocessed->points[i].y,
										  Cloudprocessed->points[i].z);
				//该点的法线
				Eigen::Vector3d pointnormcurr(cloud_normals->points[i].normal_x,
										  cloud_normals->points[i].normal_y,
										  cloud_normals->points[i].normal_z);
				
				//分别计算当前点的9个特征 并保存在对应数组中
				Eigen::Vector3d tmpcross = pointcurr.cross(pointnormcurr);
				samplefeature1[i] = (tmpcross.dot(X_axis)) * a_2d[i] * a_2d[i];
				samplefeature2[i] = -samplefeature1[i];
				samplefeature3[i] = (tmpcross.dot(Y_axis)) * a_2d[i] * a_2d[i];
				samplefeature4[i] = -samplefeature3[i];
				samplefeature5[i] = (tmpcross.dot(Z_axis)) * a_2d[i] * a_2d[i];
				samplefeature6[i] = -samplefeature5[i];
				samplefeature7[i] = fabs(pointnormcurr.dot(X_axis)) * a_2d[i] * a_2d[i];
				samplefeature8[i] = fabs(pointnormcurr.dot(Y_axis)) * a_2d[i] * a_2d[i];
				samplefeature9[i] = fabs(pointnormcurr.dot(Z_axis)) * a_2d[i] * a_2d[i];
				//当前点的索引
				cloudSortInd1[i] = i;
				cloudSortInd2[i] = i;
				cloudSortInd3[i] = i;
				cloudSortInd4[i] = i;
				cloudSortInd5[i] = i;
				cloudSortInd6[i] = i;
				cloudSortInd7[i] = i;
				cloudSortInd8[i] = i;
				cloudSortInd9[i] = i;

			}
			//对9个表进行从大到小排序
			std::sort (cloudSortInd1, cloudSortInd1 + numprocessed, comp1);
			std::sort (cloudSortInd2, cloudSortInd2 + numprocessed, comp2);
			std::sort (cloudSortInd3, cloudSortInd3 + numprocessed, comp3);
			std::sort (cloudSortInd4, cloudSortInd4 + numprocessed, comp4);
			std::sort (cloudSortInd5, cloudSortInd5 + numprocessed, comp5);
			std::sort (cloudSortInd6, cloudSortInd6 + numprocessed, comp6);
			std::sort (cloudSortInd7, cloudSortInd7 + numprocessed, comp7);
			std::sort (cloudSortInd8, cloudSortInd8 + numprocessed, comp8);
			std::sort (cloudSortInd9, cloudSortInd9 + numprocessed, comp9);

			//从model point cloud中提取现有的点
			laserCloudFromMap->clear();
			// std::cout<<"BUG1!"<<endl;
			//从地图数组中得到当前所有modle point
			for (int i = 0; i < mapsize; i++)//100
			{
				// pcl::PointCloud<PointType>::Ptr ModelPointCloud[i](new pcl::PointCloud<PointType>());//要初始化？to debug
				// pcl::PointCloud<PointType> ModelPointCloud[i];//bug 不用指针数组后 就ok了 不再需要这样的初始化

				*laserCloudFromMap += ModelPointCloud[i];//
			}
			// std::cout<<"BUG2!"<<endl;
			int laserCloudMapNum = laserCloudFromMap->points.size();

			//清空之前存储的先前帧的采样点
			//CloudSampled->clear();
			CloudSampled.reset(new pcl::PointCloud<PointType>());
			

			if(laserCloudMapNum==0)//当前map model是空的，就直接取9个列表的较大的s个点
			{
				std::cout<<"current map model is empty !"<<endl;
				for (int i = 0; i < numselect; i++)//选取每个列表中前100个点 加入采样点云中
				{
					int ind1 = cloudSortInd1[i];//值较大点的索引
					CloudSampled->push_back(Cloudprocessed->points[ind1]);
				}

				for (int i = 0; i < numselect; i++)
				{
					int ind2 = cloudSortInd2[i];//
					CloudSampled->push_back(Cloudprocessed->points[ind2]);
				}

				for (int i = 0; i < numselect; i++)
				{
					int ind3 = cloudSortInd3[i];//
					CloudSampled->push_back(Cloudprocessed->points[ind3]);
				}

				for (int i = 0; i < numselect; i++)
				{
					int ind4 = cloudSortInd4[i];//
					CloudSampled->push_back(Cloudprocessed->points[ind4]);
				}

				for (int i = 0; i < numselect; i++)
				{
					int ind5 = cloudSortInd5[i];//
					CloudSampled->push_back(Cloudprocessed->points[ind5]);
				}

				for (int i = 0; i < numselect; i++)
				{
					int ind6 = cloudSortInd6[i];//
					CloudSampled->push_back(Cloudprocessed->points[ind6]);
				}

				for (int i = 0; i < numselect; i++)
				{
					int ind7 = cloudSortInd7[i];//
					CloudSampled->push_back(Cloudprocessed->points[ind7]);
				}

				for (int i = 0; i < numselect; i++)
				{
					int ind8 = cloudSortInd8[i];//
					CloudSampled->push_back(Cloudprocessed->points[ind8]);
				}

				for (int i = 0; i < numselect; i++)
				{
					int ind9 = cloudSortInd9[i];//
					CloudSampled->push_back(Cloudprocessed->points[ind9]);
				}

				
			}
			else//否则还要判断是否是outlier
			{
				printf("points size of current map model %d \n", laserCloudMapNum);//输出当前地图的大小
				//建立model points kd tree
				kdtreeFromMap->setInputCloud(laserCloudFromMap);
				int numpicked = 0;//计数
				for (int i = 0; i < numprocessed; i++)//遍历所有点 直到找满100个点
				{
					int ind1 = cloudSortInd1[i];//值较大点的索引
					pointOri = Cloudprocessed->points[ind1];
					pointAssociateToMap(&pointOri, &pointSel);//将该采样点转为map世界坐标系  使用的是odom的结果位姿 这步转换是否需要？
					//在现有地图点中找到距离采样点最近的一个点
					kdtreeFromMap->nearestKSearch(pointSel, 1, pointSearchInd, pointSearchSqDis);
					if (sqrt(pointSearchSqDis[0]) > RadThr)//此两点之间距离太大，认为是野点 
					{
						continue;//跳过，下个点
					}
					
					CloudSampled->push_back(Cloudprocessed->points[ind1]);//注意加入的是变换前的点！
					numpicked++;
					if (numpicked >= numselect)//若已经选够100点 结束循环
					{
						break;
					}
					
				}

				numpicked = 0;//计数器清零
				for (int i = 0; i < numprocessed; i++)//遍历所有点 直到找满100个点
				{
					int ind2 = cloudSortInd2[i];//值较大点的索引
					pointOri = Cloudprocessed->points[ind2];
					pointAssociateToMap(&pointOri, &pointSel);//将该采样点转为map世界坐标系  使用的是odom的结果位姿
					//在现有地图点中找到距离采样点最近的一个点
					kdtreeFromMap->nearestKSearch(pointSel, 1, pointSearchInd, pointSearchSqDis);
					if (sqrt(pointSearchSqDis[0]) > RadThr)//此两点之间距离太大，认为是野点 
					{
						continue;//跳过，下个点
					}
					
					CloudSampled->push_back(Cloudprocessed->points[ind2]);//注意加入的是变换前的点！
					numpicked++;
					if (numpicked >= numselect)//若已经选够100点 结束循环
					{
						break;
					}	
				}

				numpicked = 0;//计数器清零
				for (int i = 0; i < numprocessed; i++)//遍历所有点 直到找满100个点
				{
					int ind3 = cloudSortInd3[i];//值较大点的索引
					pointOri = Cloudprocessed->points[ind3];
					pointAssociateToMap(&pointOri, &pointSel);//将该采样点转为map世界坐标系  使用的是odom的结果位姿
					//在现有地图点中找到距离采样点最近的一个点
					kdtreeFromMap->nearestKSearch(pointSel, 1, pointSearchInd, pointSearchSqDis);
					if (sqrt(pointSearchSqDis[0]) > RadThr)//此两点之间距离太大，认为是野点 
					{
						continue;//跳过，下个点
					}
					
					CloudSampled->push_back(Cloudprocessed->points[ind3]);//注意加入的是变换前的点！
					numpicked++;
					if (numpicked >= numselect)//若已经选够100点 结束循环
					{
						break;
					}	
				}

				numpicked = 0;//计数器清零
				for (int i = 0; i < numprocessed; i++)//遍历所有点 直到找满100个点
				{
					int ind4 = cloudSortInd4[i];//值较大点的索引
					pointOri = Cloudprocessed->points[ind4];
					pointAssociateToMap(&pointOri, &pointSel);//将该采样点转为map世界坐标系  使用的是odom的结果位姿
					//在现有地图点中找到距离采样点最近的一个点
					kdtreeFromMap->nearestKSearch(pointSel, 1, pointSearchInd, pointSearchSqDis);
					if (sqrt(pointSearchSqDis[0]) > RadThr)//此两点之间距离太大，认为是野点 
					{
						continue;//跳过，下个点
					}
					
					CloudSampled->push_back(Cloudprocessed->points[ind4]);//注意加入的是变换前的点！
					numpicked++;
					if (numpicked >= numselect)//若已经选够100点 结束循环
					{
						break;
					}	
				}

				numpicked = 0;//计数器清零
				for (int i = 0; i < numprocessed; i++)//遍历所有点 直到找满100个点
				{
					int ind5 = cloudSortInd5[i];//值较大点的索引
					pointOri = Cloudprocessed->points[ind5];
					pointAssociateToMap(&pointOri, &pointSel);//将该采样点转为map世界坐标系  使用的是odom的结果位姿
					//在现有地图点中找到距离采样点最近的一个点
					kdtreeFromMap->nearestKSearch(pointSel, 1, pointSearchInd, pointSearchSqDis);
					if (sqrt(pointSearchSqDis[0]) > RadThr)//此两点之间距离太大，认为是野点 
					{
						continue;//跳过，下个点
					}
					
					CloudSampled->push_back(Cloudprocessed->points[ind5]);//注意加入的是变换前的点！
					numpicked++;
					if (numpicked >= numselect)//若已经选够100点 结束循环
					{
						break;
					}	
				}

				numpicked = 0;//计数器清零
				for (int i = 0; i < numprocessed; i++)//遍历所有点 直到找满100个点
				{
					int ind6 = cloudSortInd6[i];//值较大点的索引
					pointOri = Cloudprocessed->points[ind6];
					pointAssociateToMap(&pointOri, &pointSel);//将该采样点转为map世界坐标系  使用的是odom的结果位姿
					//在现有地图点中找到距离采样点最近的一个点
					kdtreeFromMap->nearestKSearch(pointSel, 1, pointSearchInd, pointSearchSqDis);
					if (sqrt(pointSearchSqDis[0]) > RadThr)//此两点之间距离太大，认为是野点 
					{
						continue;//跳过，下个点
					}
					
					CloudSampled->push_back(Cloudprocessed->points[ind6]);//注意加入的是变换前的点！
					numpicked++;
					if (numpicked >= numselect)//若已经选够100点 结束循环
					{
						break;
					}	
				}

				numpicked = 0;//计数器清零
				for (int i = 0; i < numprocessed; i++)//遍历所有点 直到找满100个点
				{
					int ind7 = cloudSortInd7[i];//值较大点的索引
					pointOri = Cloudprocessed->points[ind7];
					pointAssociateToMap(&pointOri, &pointSel);//将该采样点转为map世界坐标系  使用的是odom的结果位姿
					//在现有地图点中找到距离采样点最近的一个点
					kdtreeFromMap->nearestKSearch(pointSel, 1, pointSearchInd, pointSearchSqDis);
					if (sqrt(pointSearchSqDis[0]) > RadThr)//此两点之间距离太大，认为是野点 
					{
						continue;//跳过，下个点
					}
					
					CloudSampled->push_back(Cloudprocessed->points[ind7]);//注意加入的是变换前的点！
					numpicked++;
					if (numpicked >= numselect)//若已经选够100点 结束循环
					{
						break;
					}	
				}

				numpicked = 0;//计数器清零
				for (int i = 0; i < numprocessed; i++)//遍历所有点 直到找满100个点
				{
					int ind8 = cloudSortInd8[i];//值较大点的索引
					pointOri = Cloudprocessed->points[ind8];
					pointAssociateToMap(&pointOri, &pointSel);//将该采样点转为map世界坐标系  使用的是odom的结果位姿
					//在现有地图点中找到距离采样点最近的一个点
					kdtreeFromMap->nearestKSearch(pointSel, 1, pointSearchInd, pointSearchSqDis);
					if (sqrt(pointSearchSqDis[0]) > RadThr)//此两点之间距离太大，认为是野点 
					{
						continue;//跳过，下个点
					}
					
					CloudSampled->push_back(Cloudprocessed->points[ind8]);//注意加入的是变换前的点！
					numpicked++;
					if (numpicked >= numselect)//若已经选够100点 结束循环
					{
						break;
					}	
				}

				numpicked = 0;//计数器清零
				for (int i = 0; i < numprocessed; i++)//遍历所有点 直到找满100个点
				{
					int ind9 = cloudSortInd9[i];//值较大点的索引
					pointOri = Cloudprocessed->points[ind9];
					pointAssociateToMap(&pointOri, &pointSel);//将该采样点转为map世界坐标系  使用的是odom的结果位姿
					//在现有地图点中找到距离采样点最近的一个点
					kdtreeFromMap->nearestKSearch(pointSel, 1, pointSearchInd, pointSearchSqDis);//这里是平方距离！
					if (sqrt(pointSearchSqDis[0]) > RadThr)//此两点之间距离太大，认为是野点 
					{
						continue;//跳过，下个点
					}
					
					CloudSampled->push_back(Cloudprocessed->points[ind9]);//注意加入的是变换前的点！
					numpicked++;
					if (numpicked >= numselect)//若已经选够100点 结束循环
					{
						break;
					}	
				}

			}
			printf("scan sampling time %f ms \n", t_scansample.toc());//采样总时间
			int numscansamped = CloudSampled->points.size();
			//采样前后的点数变化
			std::cout << "the size of cloud_4 is " << numprocessed << ", and the size of sampled scan is " << numscansamped << '\n';
			
			//发布采样前的点
			sensor_msgs::PointCloud2 CloudbeforeSampled;//
			pcl::toROSMsg(*Cloudprocessed, CloudbeforeSampled);
			CloudbeforeSampled.header.stamp = ros::Time().fromSec(timeLaserOdometry);
			CloudbeforeSampled.header.frame_id = "/camera_init";
			pubCloudProcessed.publish(CloudbeforeSampled);
			//for now 发布采样后的特征点
			sensor_msgs::PointCloud2 laserCloudSampled;//
			pcl::toROSMsg(*CloudSampled, laserCloudSampled);
			laserCloudSampled.header.stamp = ros::Time().fromSec(timeLaserOdometry);
			laserCloudSampled.header.frame_id = "/camera_init";
			pubCloudSampled.publish(laserCloudSampled);

			//地图中的特征点数目满足要求  若是第0帧 不执行这部分 包含匹配以及优化 to do !
			if (laserCloudMapNum > 10)
			{
				//先计算现有地图点云中的点的法线 在一次mapping中这是不变的量，用于下面求Yk
            	pcl::NormalEstimationOMP<pcl::PointXYZI, pcl::Normal> nem;
            	nem.setInputCloud (laserCloudFromMap);

            	pcl::search::KdTree<pcl::PointXYZI>::Ptr nemtree (new pcl::search::KdTree<pcl::PointXYZI>());
            	nem.setSearchMethod (nemtree);

            	pcl::PointCloud<pcl::Normal>::Ptr mapcloud_normals (new pcl::PointCloud<pcl::Normal>);
            	//设置 半径内搜索临近点
            	// nem.setRadiusSearch (0.5); //可调
				nem.setKSearch (5);//保证一定能计算有效法线 可调
            	nem.compute (*mapcloud_normals);//得到map中每个点的法线 会出现nan的情况 那是因为没有找到指定半径的邻域

				TicToc t_opt;
				TicToc t_tree;
				//建立model points kd tree
				kdtreeFromMap->setInputCloud(laserCloudFromMap);
				printf("build tree of map time %f ms \n", t_tree.toc());//建立地图点kdtree的时间
				//以下有bug出没
				//ceres优化求解 迭代20次 mapping 过程
				for (int iterCount = 0; iterCount < 20; iterCount++)
				{
					//每次迭代 都要做下面的事：

					//优化相关
					// ceres::LossFunction *loss_function = NULL;
					ceres::LossFunction *loss_function = new ceres::HuberLoss(0.1);
					ceres::LocalParameterization *q_parameterization =
						new ceres::EigenQuaternionParameterization();
					ceres::Problem::Options problem_options;

					ceres::Problem problem(problem_options);
					//要去优化的目标？
					problem.AddParameterBlock(parameters, 4, q_parameterization);
					problem.AddParameterBlock(parameters + 4, 3);
	
					//初始化点云，存储x投影的结果Yk
					// pcl::PointCloud<PointType>::Ptr CloudProjected(new pcl::PointCloud<PointType>());
					//遍历每个采样点x，得到I_x和对应的投影点y
					for (int i = 0; i < numscansamped; i++)
					{
						pointOri = CloudSampled->points[i];
						pointAssociateToMap(&pointOri, &pointSel);//将该特征点进行变换  使用的是当前？位姿 这步转换是否需要？如何同步地更新优化后的q_w_curr
						//在现有地图点中找到距离采样点不大于0.20m 的点
						kdtreeFromMap->radiusSearch(pointSel, RadThr, pointSearchInd, pointSearchSqDis);
						int numBox = pointSearchInd.size();//得到的指定半径内近邻个数
						// printf("num of Box %d \n", numBox);
						//std::vector<Eigen::Vector3d> Boxpoints;//存储若干近邻点
						std::vector<double> Wx;//权值数组
						std::vector<double> Ixi;//分子中的一项
						//pointsel x_i *
						Eigen::Vector3d pointx(pointSel.x, pointSel.y, pointSel.z);
						
						Eigen::Vector3d x_i(pointOri.x, pointOri.y, pointOri.z);
						//最近点的法线n_j
						Eigen::Vector3d nj(mapcloud_normals->points[ pointSearchInd[0] ].normal_x,
										  				mapcloud_normals->points[ pointSearchInd[0] ].normal_y,
										  				mapcloud_normals->points[ pointSearchInd[0] ].normal_z);
						if(std::isnan(nj.x()))//
						{
							std::cout <<"nj NaN Warning!"<<endl;
							std::cout << "nj: " << nj << '\n';//就是因为nj是nan
							printf("nj NaN occur at %d in sampled points in frame %d \n",i ,frameCount);
							printf("Skip this residua of xi \n");
							continue;//不把该点的残差计入总loss
							
						}
					
						for (int j = 0; j < numBox; j++)
						{	//当前来自地图中的点p_
							Eigen::Vector3d pcurrbox(laserCloudFromMap->points[ pointSearchInd[j] ].x,
										  			laserCloudFromMap->points[ pointSearchInd[j] ].y,
										  			laserCloudFromMap->points[ pointSearchInd[j] ].z);
							//当前来自地图中的点p_的法线
							Eigen::Vector3d normpcurr(mapcloud_normals->points[ pointSearchInd[j] ].normal_x,
										  				mapcloud_normals->points[ pointSearchInd[j] ].normal_y,
										  				mapcloud_normals->points[ pointSearchInd[j] ].normal_z);
							//当前点对应的权值
							double w_j = exp(-pointSearchSqDis[j]/(h_imls*h_imls));
							
							
							//to debug
							if (std::isnan(normpcurr.x()))//tmp1
							{
								std::cout <<"1NaN Warning!"<<endl;//还是会出现
								
								printf("1NaN the %d item of vector Ixi at %d in sampled points in frame %d \n",j ,i ,frameCount);
								continue;//若遇到nan跳过这个点p，也不把这里的wi,Ixi计算入内
							}
							double tmp1 = w_j*((pointx-pcurrbox).dot(normpcurr));//某一项会有nan？
							Wx.push_back(w_j);
							Ixi.push_back(tmp1);
						}
						// printf("bug876 \n");//以上都ok
						//计算采样点x到map点隐式平面的距离
						double fenzi = std::accumulate(Ixi.begin(), Ixi.end(), 0.000001);//出现了负值？合理吗
						// printf("fenzi %f \n", fenzi);//分子首先有nan！
						double fenmu = std::accumulate(Wx.begin(), Wx.end(), 0.000001);
						// printf("fenmu %f \n", fenmu);
						double I_xi = fenzi/fenmu;//会出现NaN
						// printf("I_xi %f \n", I_xi);
						// float I_xi = std::accumulate(Ix1.begin(), Ix1.end(), 0)/std::accumulate(Wx.begin(), Wx.end(), 0);//bug!
						// printf("bug884 \n");
						//x_i对应的点y_i
						Eigen::Vector3d y_i = pointx - I_xi * nj;
						// PointType pointy;
						// pointy.x = y_i.x();
        				// pointy.y = y_i.y();
        				// pointy.z = y_i.z();
						// pointy.intensity = pointSel.intensity;
						// CloudProjected->push_back(pointy);
						// printf("bug874 \n");
						//接下来本质就是 point to plane ICP了
						
						if(std::isnan(y_i.x()))//
						{
							std::cout <<"2NaN Warning!"<<endl;
							// std::cout << "xi: " << x_i << '\n';
							printf("I_xi: %f \n", I_xi);//
							// std::cout << "nj: " << nj << '\n';//就是因为nj是nan
							// std::cout << "yi: " << y_i << '\n';//若I是nan，则yi都也是nan
							printf("2NaN occur at %d in sampled points in frame %d \n",i ,frameCount);
							printf("Skip this residua of xi \n");
							continue;//不把该点的残差计入总loss
							
						}
						
						ceres::CostFunction *cost_function = LidarPoint2PlaneICP::Create(x_i, y_i, nj);
						problem.AddResidualBlock(cost_function, loss_function, parameters, parameters + 4);
						
						
					}
					// printf("bug881 \n");
					//求解优化
					TicToc t_solver;
					ceres::Solver::Options options;
					options.linear_solver_type = ceres::DENSE_QR;//属于列文伯格-马夸尔特方法
					options.max_num_iterations = 4;//一次优化的最大迭代次数
					options.minimizer_progress_to_stdout = false;//输出到cout 
					options.check_gradients = false;
					options.gradient_check_relative_precision = 1e-4;
					ceres::Solver::Summary summary;
					ceres::Solve(options, &problem, &summary);//开始优化
					printf("the %d mapping solver time %f ms \n",iterCount , t_solver.toc());

					//输出一次mapping优化得到的位姿 w x y z 当前帧相对于map world的变换 
					// printf("result q %f %f %f %f result t %f %f %f\n", parameters[3], parameters[0], parameters[1], parameters[2],
					// 	   parameters[4], parameters[5], parameters[6]);
	
				}
				printf("\nthe frame %d mapping optimization time %f \n", frameCount, t_opt.toc());
				//20次优化后的该帧位姿最后结果 与948的值是一样的
				std::cout<<"the final result of frame"<< frameCount <<": q_w_curr= "<< q_w_curr.coeffs().transpose() <<"\nt_w_curr= "<< t_w_curr.transpose() <<"\n"<<endl;
			}
			else
			{//点太少
				ROS_WARN("time map model points num are not enough");
			}
			
			//迭代优化结束 更新相关的转移矩阵
			transformUpdate(); //更新了odo world 相对于 map world的变换

			//先跳过 还没有匹配
			TicToc t_add;
			//将当前帧的点加入到modelpoint 中 相应位置
			if (frameCount<100)//说明当前model point 还没存满 直接添加
			{
				for (int i = 0; i < numprocessed; i++)//把当前优化过的scan所有点注册到地图数组指针中
				{	//将该点转移到世界坐标系下
					pointAssociateToMap(&Cloudprocessed->points[i], &pointSel);
					ModelPointCloud[frameCount].push_back(pointSel);
				}

			}
			else//当前model point数组已填满100帧 去除第一个，从后面添加新的
			{
				for (int j = 0; j < mapsize-1; j++)
				{
					pcl::PointCloud<PointType>::Ptr tmpCloud(new pcl::PointCloud<PointType>());
					*tmpCloud = ModelPointCloud[j+1];
					int numtmpCloud = tmpCloud->points.size();
					//把数组中依次前移
					ModelPointCloud[j].clear();//->
					// ModelPointCloud[j].reset(new pcl::PointCloud<PointType>());
					for (int k = 0; k < numtmpCloud; k++)
					{
						ModelPointCloud[j].push_back(tmpCloud->points[k]);
					}
					
					// ModelPointCloud[j] = ModelPointCloud[j+1];
				}
				// ModelPointCloud[mapsize-1].reset(new pcl::PointCloud<PointType>());
				ModelPointCloud[mapsize-1].clear();//->
				//把当前帧的点注册后加入到数组最后一个位置
				for (int i = 0; i < numprocessed; i++)//把当前优化过的scan所有点注册到地图数组指针中
				{	//将该点转移到世界坐标系下
					pointAssociateToMap(&Cloudprocessed->points[i], &pointSel);
					ModelPointCloud[mapsize-1].push_back(pointSel);
				}
				
			}
			printf("add points time %f ms\n", t_add.toc());

			//*/

			// /*
			TicToc t_pub;
			//publish surround map for every 5 frame
			if (frameCount % 5 == 0)
			{
				laserCloudSurround->clear();
				for (int i = 95; i < mapsize; i++)//只看最近的5帧
				{
					*laserCloudSurround += ModelPointCloud[i];
					
				}

				sensor_msgs::PointCloud2 laserCloudSurround3;
				pcl::toROSMsg(*laserCloudSurround, laserCloudSurround3);
				laserCloudSurround3.header.stamp = ros::Time().fromSec(timeLaserOdometry);
				laserCloudSurround3.header.frame_id = "/camera_init";
				pubLaserCloudSurround.publish(laserCloudSurround3);
			}

			//每隔20帧发布一次整个特征点地图 （21,21,11=）4851个cube中的点  上面的是局部的周围cube 85个
			//model map 的点
			if (frameCount % 20 == 0)
			{
				pcl::PointCloud<PointType> laserCloudMap;
				for (int i = 85; i < mapsize; i++)//100帧的地图点太多了 看后15帧
				{
					laserCloudMap += ModelPointCloud[i];
				}
				sensor_msgs::PointCloud2 laserCloudMsg;
				pcl::toROSMsg(laserCloudMap, laserCloudMsg);
				laserCloudMsg.header.stamp = ros::Time().fromSec(timeLaserOdometry);
				laserCloudMsg.header.frame_id = "/camera_init";
				pubLaserCloudMap.publish(laserCloudMsg);
			}

			//将点云中全部点转移到世界坐标系下  之前的mapping只使用了特征点
			int laserCloudFullResNum = laserCloudFullRes->points.size();
			for (int i = 0; i < laserCloudFullResNum; i++)
			{
				pointAssociateToMap(&laserCloudFullRes->points[i], &laserCloudFullRes->points[i]);
			}

			sensor_msgs::PointCloud2 laserCloudFullRes3;//当前帧的所有点  多于Cloudprocessed
			pcl::toROSMsg(*laserCloudFullRes, laserCloudFullRes3);
			laserCloudFullRes3.header.stamp = ros::Time().fromSec(timeLaserOdometry);
			laserCloudFullRes3.header.frame_id = "/camera_init";
			pubLaserCloudFullRes.publish(laserCloudFullRes3);///velodyne_cloud_registered 当前帧已注册的点

			printf("mapping pub time %f ms \n", t_pub.toc());

			// */

			//整个mapping的用时
			printf("whole mapping time %f ms **************************\n \n", t_whole.toc());

			//将结果位姿转为kitti benchmark(camera)参考系
			Eigen::Matrix3d R_correct;
    		R_correct << 0, -1, 0, 0, 0, -1, 1, 0, 0; // 坐标系的一个旋转
    		Eigen::Quaterniond q_correct(R_correct); //对应的四元数
    		// cout << "q_correct: " << q_correct.coeffs().transpose() <<endl; //输出该旋转四元数: x,y,z,w 0.5 -0.5 0.5 0.5

			// Eigen::Quaterniond q_w_curr_correct = q_correct * q_w_curr;//没必要对选装四元数做变换，因为最后计算旋转误差只看转角
			Eigen::Quaterniond q_w_curr_correct = q_w_curr; //这样证明结果最优 rpy对应上了 在记录结果是再去变换
			q_w_curr_correct.normalize();
			Eigen::Vector3d t_w_curr_correct = q_correct * t_w_curr;

			// /*
			nav_msgs::Odometry odomAftMapped;
			odomAftMapped.header.frame_id = "/camera_init";
			odomAftMapped.child_frame_id = "/aft_mapped";
			odomAftMapped.header.stamp = ros::Time().fromSec(timeLaserOdometry);
			odomAftMapped.pose.pose.orientation.x = q_w_curr_correct.x();
			odomAftMapped.pose.pose.orientation.y = q_w_curr_correct.y();
			odomAftMapped.pose.pose.orientation.z = q_w_curr_correct.z();
			odomAftMapped.pose.pose.orientation.w = q_w_curr_correct.w();
			odomAftMapped.pose.pose.position.x = t_w_curr_correct(0);
			odomAftMapped.pose.pose.position.y = t_w_curr_correct(1);
			odomAftMapped.pose.pose.position.z = t_w_curr_correct(2);
			pubOdomAftMapped.publish(odomAftMapped);//注意这个和前面的highfrequency不一样

			geometry_msgs::PoseStamped laserAfterMappedPose;
			laserAfterMappedPose.header = odomAftMapped.header;
			laserAfterMappedPose.pose = odomAftMapped.pose.pose;
			laserAfterMappedPath.header.stamp = odomAftMapped.header.stamp;
			laserAfterMappedPath.header.frame_id = "/camera_init";
			laserAfterMappedPath.poses.push_back(laserAfterMappedPose);
			pubLaserAfterMappedPath.publish(laserAfterMappedPath);

			//广播坐标系的变换 这个作用和上面的位姿不重复吗？
			static tf::TransformBroadcaster br;
			tf::Transform transform;
			tf::Quaternion q;
			transform.setOrigin(tf::Vector3(t_w_curr_correct(0),
											t_w_curr_correct(1),
											t_w_curr_correct(2)));
			q.setW(q_w_curr_correct.w());
			q.setX(q_w_curr_correct.x());
			q.setY(q_w_curr_correct.y());
			q.setZ(q_w_curr_correct.z());
			transform.setRotation(q);
			br.sendTransform(tf::StampedTransform(transform, odomAftMapped.header.stamp, "/camera_init", "/aft_mapped"));
			// */

			frameCount++;
		}
		std::chrono::milliseconds dura(2);
        std::this_thread::sleep_for(dura);
	}
}


int main(int argc, char **argv)
{
	ros::init(argc, argv, "laserMapping");
	ros::NodeHandle nh;

	// float lineRes = 0;
	// float planeRes = 0;
	// nh.param<float>("mapping_line_resolution", lineRes, 0.4); //aloam_velodyne_HDL_64.launch中的参数设置
	// nh.param<float>("mapping_plane_resolution", planeRes, 0.8);
	// printf("line resolution %f plane resolution %f \n", lineRes, planeRes);
	//设置体素栅格滤波器 体素大小
	// downSizeFilterCorner.setLeafSize(lineRes, lineRes,lineRes);
	// downSizeFilterSurf.setLeafSize(planeRes, planeRes, planeRes);

	//接收来自laserodo.cpp发来的处理后的点云/cloud_4 topic
	ros::Subscriber subCloudprocessed = nh.subscribe<sensor_msgs::PointCloud2>("/cloud_4", 100, CloudprocessedHandler);

	ros::Subscriber subLaserCloudCornerLast = nh.subscribe<sensor_msgs::PointCloud2>("/laser_cloud_corner_last", 100, laserCloudCornerLastHandler);

	ros::Subscriber subLaserCloudSurfLast = nh.subscribe<sensor_msgs::PointCloud2>("/laser_cloud_surf_last", 100, laserCloudSurfLastHandler);

	ros::Subscriber subLaserOdometry = nh.subscribe<nav_msgs::Odometry>("/laser_odom_to_init", 100, laserOdometryHandler);

	ros::Subscriber subLaserCloudFullRes = nh.subscribe<sensor_msgs::PointCloud2>("/velodyne_cloud_3", 100, laserCloudFullResHandler);

	//参数中的100 ：排队等待处理的传入消息数（超出此队列容量的消息将被丢弃）

	pubCloudProcessed = nh.advertise<sensor_msgs::PointCloud2>("/cloud_before_sampled", 100);//发布当前采样前的点 和cloud4一样
 
	pubCloudSampled = nh.advertise<sensor_msgs::PointCloud2>("/laser_cloud_sampled", 100);//发布当前采样后的点

	pubLaserCloudSurround = nh.advertise<sensor_msgs::PointCloud2>("/laser_cloud_surround", 100);//周围的地图点

	pubLaserCloudMap = nh.advertise<sensor_msgs::PointCloud2>("/laser_cloud_map", 100);//更多的地图点

	pubLaserCloudFullRes = nh.advertise<sensor_msgs::PointCloud2>("/velodyne_cloud_registered", 100);//当前帧（已注册）的所有点

	pubOdomAftMapped = nh.advertise<nav_msgs::Odometry>("/aft_mapped_to_init", 100);//优化后的位姿？

	//pubOdomAftMappedHighFrec = nh.advertise<nav_msgs::Odometry>("/aft_mapped_to_init_high_frec", 100);//接收odo的位姿 高频输出 不是最后的优化结果

	pubLaserAfterMappedPath = nh.advertise<nav_msgs::Path>("/aft_mapped_path", 100);

	//mapping过程 单开一个线程
	std::thread mapping_process{imls};

	ros::spin();

	return 0;
}