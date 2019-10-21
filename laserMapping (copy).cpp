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
#include <stdlib.h>
#include <time.h> 

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

int frameCount = 0; // 计数！
bool systemInited = false;
//10.18 update:索性从头就不再计算LOAM那些多余的结果了
const int systemDelay_map = 2;//正常情况下只有前3帧才依靠odo传送的位姿进行mapping :6，3，4662(max,就是全部LOAM计算保留了) 就要前两帧  0 意味着全靠IMLS

//时间戳
double timecloudprocessed = 0;
double timecloudGround = 0;
double timeLaserCloudCornerLast = 0;
double timeLaserCloudSurfLast = 0;
double timeLaserCloudFullRes = 0;
double timeLaserOdometry = 0;

//keep n =100 scan in the map
#define mapsize 100
//从每个scan list中采样的点数s=100
#define numselect 100
//采样和匹配中的 近邻距离阈值
#define SampRadThr 0.20  //0.2；再调的参数
#define SampNeiThr 0.20 //计算特征点法线 的近邻范围 原来是0.5
#define RadThr 0.20 //计算I_x时那个box邻域半径
#define numkneibor 5 //计算map 点法线 时近邻个数 5，8，9
//定义IMLS surface的参数 h
#define h_imls 0.06
//lossfunction 的阈值参数
#define lossarg 0.1 //huberloss 原始是0.1 0.2 
//ICP优化次数
#define numICP 20 //论文是20次
// input: from laserodometry.cpp  接收到的边沿点和平面点
pcl::PointCloud<PointType>::Ptr laserCloudCornerLast(new pcl::PointCloud<PointType>());
pcl::PointCloud<PointType>::Ptr laserCloudSurfLast(new pcl::PointCloud<PointType>());

//当前帧采样之后用于maping的点  9*100=900
pcl::PointCloud<PointType>::Ptr CloudSampled(new pcl::PointCloud<PointType>());


//input & output: points in one frame. local --> global 一帧的所有点
pcl::PointCloud<PointType>::Ptr laserCloudFullRes(new pcl::PointCloud<PointType>());
pcl::PointCloud<PointType>::Ptr Cloudprocessed(new pcl::PointCloud<PointType>());
pcl::PointCloud<PointType>::Ptr CloudGroundCurr(new pcl::PointCloud<PointType>());
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

//当前帧k相对于 odom world的变换
Eigen::Quaterniond q_wodom_curr(1, 0, 0, 0);
Eigen::Vector3d t_wodom_curr(0, 0, 0);
// 接受odo结果 作为备份
Eigen::Quaterniond q_odom_b(1, 0, 0, 0);
Eigen::Vector3d t_odom_b(0, 0, 0);

//线性插值需要的
//第k-1帧的相对于 odom world的变换
Eigen::Quaterniond q_wodom_k_1(1, 0, 0, 0);
Eigen::Vector3d t_wodom_k_1(0, 0, 0);
//第k-2帧相对于 odom world的变换
Eigen::Quaterniond q_wodom_k_2(1, 0, 0, 0);
Eigen::Vector3d t_wodom_k_2(0, 0, 0);
//第k-1帧的相对于  world的变换
Eigen::Quaterniond q_w_k_1(1, 0, 0, 0);
Eigen::Vector3d t_w_k_1(0, 0, 0);
//第k-2帧相对于  world的变换
Eigen::Quaterniond q_w_k_2(1, 0, 0, 0);
Eigen::Vector3d t_w_k_2(0, 0, 0);

//缓存来自laserodometry.cpp的量
std::queue<sensor_msgs::PointCloud2ConstPtr> processedBuf; //缓存cloud_4的
std::queue<sensor_msgs::PointCloud2ConstPtr> GroundLastBuf; //缓存地面点的
std::queue<sensor_msgs::PointCloud2ConstPtr> cornerLastBuf;
std::queue<sensor_msgs::PointCloud2ConstPtr> surfLastBuf;
std::queue<sensor_msgs::PointCloud2ConstPtr> fullResBuf;
std::queue<nav_msgs::Odometry::ConstPtr> odometryBuf;
std::mutex mBuf;

//创建voxelgrid滤波器 （体素栅格滤波器）
// pcl::VoxelGrid<PointType> downSizeFilterCorner;
pcl::VoxelGrid<PointType> downSizeFilterSurf;// 用于下采样地面特征点

std::vector<int> pointSearchInd;
std::vector<float> pointSearchSqDis;
std::vector<int> pointNeighborInd; 
std::vector<float> pointNeighborSqDis;


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
//应对线性估计的 q_wodom_curr 太离谱的情况
void transformAssociateToMap_b()
{
	q_w_curr = q_wmap_wodom * q_odom_b;
	t_w_curr = q_wmap_wodom * t_odom_b + t_wmap_wodom;
}

//
void transformUpdate_b()
{// transformAssociateToMap() 的逆过程？
	q_wmap_wodom = q_w_curr * q_odom_b.inverse();
	t_wmap_wodom = t_w_curr - q_wmap_wodom * t_odom_b;
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
//处理传来的地面点
void CloudGroundLastHandler(const sensor_msgs::PointCloud2ConstPtr &CloudGroundLast2)
{
	mBuf.lock();
	GroundLastBuf.push(CloudGroundLast2);
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
		//只利用第2帧的odom结果拿来作为一个参考 该版本匹配部分还是IMLS
		if(frameCount < systemDelay_map)//前10帧仍需要先前的节点输出 与之前都一样
		{
			// while (!cornerLastBuf.empty() && !surfLastBuf.empty() && //确保接收到laserodo.cpp发出的点云
			// 	!fullResBuf.empty()  && !processedBuf.empty() && !odometryBuf.empty() && !GroundLastBuf.empty() )
			while (!fullResBuf.empty()  && !processedBuf.empty() && !odometryBuf.empty() && !GroundLastBuf.empty() )
			{
				mBuf.lock();
				//确保各容器数据的时间戳是合理的
				while (!odometryBuf.empty() && odometryBuf.front()->header.stamp.toSec() < fullResBuf.front()->header.stamp.toSec())
					odometryBuf.pop();
				if (odometryBuf.empty())
				{
					mBuf.unlock();
					break;
				}
				/*
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
				*/
				//待采样的cloud
				while (!processedBuf.empty() && processedBuf.front()->header.stamp.toSec() < fullResBuf.front()->header.stamp.toSec())
					processedBuf.pop();
				if (processedBuf.empty())
				{
					mBuf.unlock();
					break;//无数据，重新接收，并执行判断是否接收到
				}

				while (!GroundLastBuf.empty() && GroundLastBuf.front()->header.stamp.toSec() < fullResBuf.front()->header.stamp.toSec())
					GroundLastBuf.pop();
				if (GroundLastBuf.empty())
				{
					mBuf.unlock();
					break;//无数据，重新接收，并执行判断是否接收到
				}

				//得到时间戳
				timecloudprocessed = processedBuf.front()->header.stamp.toSec();
				timecloudGround = GroundLastBuf.front()->header.stamp.toSec();
				// timeLaserCloudCornerLast = cornerLastBuf.front()->header.stamp.toSec();
				// timeLaserCloudSurfLast = surfLastBuf.front()->header.stamp.toSec();
				timeLaserCloudFullRes = fullResBuf.front()->header.stamp.toSec();
				timeLaserOdometry = odometryBuf.front()->header.stamp.toSec();

				//确保各类点云的时间戳同步（那前面为啥也判断了时间？）
				if (/*timeLaserCloudCornerLast != timeLaserOdometry ||
					timeLaserCloudSurfLast != timeLaserOdometry ||*/
					timeLaserCloudFullRes != timeLaserOdometry || 
					timecloudprocessed != timeLaserOdometry ||
					timecloudGround != timeLaserOdometry)
				{
					printf("time full %f odom %f processed %f \n", timeLaserCloudFullRes, timeLaserOdometry, timecloudprocessed);
					printf("unsync messeage!");
					mBuf.unlock();
					break;
				}

				//数据从容器到点云指针
				// laserCloudCornerLast->clear();
				// pcl::fromROSMsg(*cornerLastBuf.front(), *laserCloudCornerLast);
				// cornerLastBuf.pop();

				// laserCloudSurfLast->clear();
				// pcl::fromROSMsg(*surfLastBuf.front(), *laserCloudSurfLast);
				// surfLastBuf.pop();

				laserCloudFullRes->clear();
				pcl::fromROSMsg(*fullResBuf.front(), *laserCloudFullRes);
				fullResBuf.pop();

				Cloudprocessed->clear();
				pcl::fromROSMsg(*processedBuf.front(), *Cloudprocessed);
				processedBuf.pop();

				CloudGroundCurr->clear();
				pcl::fromROSMsg(*GroundLastBuf.front(), *CloudGroundCurr);
				GroundLastBuf.pop();

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
				// pcl::NormalEstimationOMP<pcl::PointXYZI, pcl::Normal> ne;
				// ne.setInputCloud (Cloudprocessed);

				// pcl::search::KdTree<pcl::PointXYZI>::Ptr netree (new pcl::search::KdTree<pcl::PointXYZI>());
				// ne.setSearchMethod (netree);

				// pcl::PointCloud<pcl::Normal>::Ptr cloud_normals (new pcl::PointCloud<pcl::Normal>);
				// //设置 半径内搜索临近点
				// ne.setRadiusSearch (0.5);
				// ne.compute (*cloud_normals);//得到当前scan：Cloudprocessed中每个点的法线
				
				//但是还要得到法线计算过程中的中间值 PCA的特征值
				kdtreeProcessed->setInputCloud(Cloudprocessed);
				
				int numprocessed = Cloudprocessed->points.size();
				double a_2d[numprocessed];//保存每个点的planar scalar
				
				//遍历每点计算a2d以及9个特征
				for (int i=0; i<numprocessed; i++)
				{
					pointSel = Cloudprocessed->points[i];
					//对于当前点在scan中搜索指定半径内的近邻  近邻半径可调
					kdtreeProcessed->radiusSearch(pointSel, SampNeiThr, pointSearchInd, pointSearchSqDis);
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
					//注意这里还有一个正则因子！ 修正！
					covMat = covMat * (1/double(numneighbor));
					Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> saes(covMat);//协方差矩阵特征值分解
					// note Eigen library sort eigenvalues in increasing order
					//得到三个特征值 从大到小
					double lamda_1 = sqrt(saes.eigenvalues()[2]);
					double lamda_2 = sqrt(saes.eigenvalues()[1]);
					double lamda_3 = sqrt(saes.eigenvalues()[0]);

					//获取最小特征值对应的特征向量 即为法线
					Eigen::Vector3d rawnormcurr = saes.eigenvectors().col(0);
					//计算当前点近邻的planar scale
					a_2d[i] = (lamda_2 - lamda_3)/lamda_1;

					Eigen::Vector3d X_axis(1, 0, 0);
					Eigen::Vector3d Y_axis(0, 1, 0);
					Eigen::Vector3d Z_axis(0, 0, 1);

					Eigen::Vector3d pointcurr(Cloudprocessed->points[i].x,
											Cloudprocessed->points[i].y,
											Cloudprocessed->points[i].z);
					//该点的法线
					// Eigen::Vector3d pointnormcurr(cloud_normals->points[i].normal_x,
					// 						cloud_normals->points[i].normal_y,
					// 						cloud_normals->points[i].normal_z);

					Eigen::Vector3d pointnormcurr = rawnormcurr.normalized();//归一化
					
					
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
				
				//先对地面点进行下采样吧，否则点太多了
				// pcl::PointCloud<PointType>::Ptr CloudGroundDS(new pcl::PointCloud<PointType>());
				// downSizeFilterSurf.setInputCloud(CloudGroundCurr);
				// downSizeFilterSurf.filter(*CloudGroundDS);
				// int numGroundDS = CloudGroundDS->points.size();
				

				if(laserCloudMapNum==0)//当前map model是空的，就直接取9个列表的较大的s个点
				{
					std::cout<<"current map model is empty !"<<endl;
					
					// for (int i = 0; i < numGroundDS; i++)//选取所有地面点
					// {
						
					// 	CloudSampled->push_back(CloudGroundDS->points[i]);
					// }
					
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
					
					// for (int i = 0; i < numGroundDS; i++)//遍历所有地面点 ,只要非野点都要
					// {
					// 	pointOri = CloudGroundDS->points[i];
					// 	pointAssociateToMap(&pointOri, &pointSel);//将该采样点转为map世界坐标系  使用的是odom的结果位姿 这步转换是否需要？
					// 	//在现有地图点中找到距离采样点最近的一个点
					// 	kdtreeFromMap->nearestKSearch(pointSel, 1, pointSearchInd, pointSearchSqDis);
					// 	if (sqrt(pointSearchSqDis[0]) > SampRadThr)//此两点之间距离太大，认为是野点 
					// 	{
					// 		continue;//跳过，下个点
					// 	}
						
					// 	CloudSampled->push_back(CloudGroundDS->points[i]);//注意加入的是变换前的点！
					// 	// numpicked++;
					// 	// if (numpicked >= numselect)//若已经选够100点 结束循环
					// 	// {
					// 	// 	break;
					// 	// }
					// }
					
					int numpicked = 0;//计数
					for (int i = 0; i < numprocessed; i++)//遍历所有点 直到找满100个点
					{
						int ind1 = cloudSortInd1[i];//值较大点的索引
						pointOri = Cloudprocessed->points[ind1];
						pointAssociateToMap(&pointOri, &pointSel);//将该采样点转为map世界坐标系  使用的是odom的结果位姿 这步转换是否需要？
						//在现有地图点中找到距离采样点最近的一个点
						kdtreeFromMap->nearestKSearch(pointSel, 1, pointSearchInd, pointSearchSqDis);
						if (sqrt(pointSearchSqDis[0]) > SampRadThr)//此两点之间距离太大，认为是野点 
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
						if (sqrt(pointSearchSqDis[0]) > SampRadThr)//此两点之间距离太大，认为是野点 
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
						if (sqrt(pointSearchSqDis[0]) > SampRadThr)//此两点之间距离太大，认为是野点 
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
						if (sqrt(pointSearchSqDis[0]) > SampRadThr)//此两点之间距离太大，认为是野点 
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
						if (sqrt(pointSearchSqDis[0]) > SampRadThr)//此两点之间距离太大，认为是野点 
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
						if (sqrt(pointSearchSqDis[0]) > SampRadThr)//此两点之间距离太大，认为是野点 
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
						if (sqrt(pointSearchSqDis[0]) > SampRadThr)//此两点之间距离太大，认为是野点 
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
						if (sqrt(pointSearchSqDis[0]) > SampRadThr)//此两点之间距离太大，认为是野点 
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
						if (sqrt(pointSearchSqDis[0]) > SampRadThr)//此两点之间距离太大，认为是野点 
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
				
				// if(CloudSampled->points.size()< 900)
				// {
				// 	printf("WARNING: the frame %d has 0 sampled points ! \n", frameCount);
				// }
				// else
				// {
				// 	std::cout << "the size of sampled points of frame " << frameCount << " is " << CloudSampled->points.size() << '\n';
				// }
				
				int numscansamped = CloudSampled->points.size();//使用估计的位姿，会出现采样点为0的情况
				//采样前后的点数变化  正常是采到900点 多出的就是地面点  (with dsground)
				std::cout << "the size before sampled : " << numprocessed << " and the size after sampled is " << numscansamped << '\n';
				
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
					//不这样一下子就计算所有点的法线 而是每次只计算需要的地图中的近邻点的法线
					//先计算现有地图点云中的点的法线 在一次mapping中这是不变的量，用于下面求Yk
					
					// pcl::NormalEstimationOMP<pcl::PointXYZI, pcl::Normal> nem;
					// nem.setInputCloud (laserCloudFromMap);

					// pcl::search::KdTree<pcl::PointXYZI>::Ptr nemtree (new pcl::search::KdTree<pcl::PointXYZI>());
					// nem.setSearchMethod (nemtree);

					// pcl::PointCloud<pcl::Normal>::Ptr mapcloud_normals (new pcl::PointCloud<pcl::Normal>);
					// //设置 半径内搜索临近点
					// // nem.setRadiusSearch (0.5); //可调
					// nem.setKSearch (numkneibor);//保证一定能计算有效法线 可调 5,9
					// nem.compute (*mapcloud_normals);//得到map中每个点的法线 会出现nan的情况 那是因为没有找到指定半径的邻域
					
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
							
							Eigen::Vector3d x_i(pointOri.x, pointOri.y, pointOri.z);//进入优化器中

							//最近点的法线n_j 论文的意思是只需计算最近点法线，其他box近邻内的法线值去一样的值近似！
							PointType nearestp = laserCloudFromMap->points[ pointSearchInd[0] ];
							Eigen::Vector3d nearestv(laserCloudFromMap->points[ pointSearchInd[0] ].x,
													laserCloudFromMap->points[ pointSearchInd[0] ].y,
													laserCloudFromMap->points[ pointSearchInd[0] ].z);
							
							//寻找地图点中关于nearestp的邻域 拟合平面 计算法线
							kdtreeFromMap->radiusSearch(nearestp, SampNeiThr, pointNeighborInd, pointNeighborSqDis);
							int numneighborm = pointNeighborInd.size();//得到的半径内近邻个数
							std::vector<Eigen::Vector3d> neighborsm;//存储若干近邻点
							Eigen::Vector3d centerm(0, 0, 0);//初始化近邻点的重心
							for (int j = 0; j < numneighborm; j++)
							{
								Eigen::Vector3d tmpm(laserCloudFromMap->points[pointNeighborInd[j]].x,
													laserCloudFromMap->points[pointNeighborInd[j]].y,
													laserCloudFromMap->points[pointNeighborInd[j]].z);
								centerm = centerm + tmpm;
								neighborsm.push_back(tmpm);
							}
							//得到近邻点坐标的重心
							centerm = centerm / double(numneighborm);

							Eigen::Matrix3d covMatm = Eigen::Matrix3d::Zero();//近邻点的协方差矩阵
							for (int j = 0; j < numneighborm; j++)
							{
								Eigen::Matrix<double, 3, 1> tmpZeroMeanm = neighborsm[j] - centerm;
								covMatm = covMatm + tmpZeroMeanm * tmpZeroMeanm.transpose();
							}
							//注意这里还有一个正则因子！ 修正！
							covMatm = covMatm * (1/double(numneighborm));
							Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> saesm(covMatm);//协方差矩阵特征值分解
							// note Eigen library sort eigenvalues in increasing order

							//获取最小特征值对应的特征向量 即为法线
							Eigen::Vector3d rawnorm = saesm.eigenvectors().col(0);

							// Eigen::Vector3d nj(mapcloud_normals->points[ pointSearchInd[0] ].normal_x,
							// 								mapcloud_normals->points[ pointSearchInd[0] ].normal_y,
							// 								mapcloud_normals->points[ pointSearchInd[0] ].normal_z);
							if(std::isnan(rawnorm.x()))//这和估计map model中每个点的法线有关
							{
								std::cout <<"nj NaN Warning!"<<endl;
								// std::cout << "nj: " << rawnorm << '\n';//就是因为nj是nan
								printf("nj NaN occur at %d th sampled points in frame %d @ %d iteration\n",i ,frameCount, iterCount);
								printf("Skip this residua of xi \n");
								continue;//不把该点的残差计入总loss
							}

							Eigen::Vector3d nj = rawnorm.normalized();//归一化
							if( nearestv.dot(nj) > 0)
							{//法线方向统一
								nj = -nj;
							}
						
							for (int j = 0; j < numBox; j++)
							{	//当前来自地图中的点p_
								Eigen::Vector3d pcurrbox(laserCloudFromMap->points[ pointSearchInd[j] ].x,
														laserCloudFromMap->points[ pointSearchInd[j] ].y,
														laserCloudFromMap->points[ pointSearchInd[j] ].z);
								//当前来自地图中的点p_的法线
								// Eigen::Vector3d normpcurr(mapcloud_normals->points[ pointSearchInd[j] ].normal_x,
								// 							mapcloud_normals->points[ pointSearchInd[j] ].normal_y,
								// 							mapcloud_normals->points[ pointSearchInd[j] ].normal_z);
								Eigen::Vector3d normpcurr = nj;
								// //当前点对应的权值
								double w_j = exp(-pointSearchSqDis[j]/(h_imls*h_imls));
								
								
								//to debug
								// if (std::isnan(normpcurr.x()))//tmp1
								// {
								// 	std::cout <<"iNaN Warning!"<<endl;//还是会出现
									
								// 	printf("iNaN the %d item of vector Ixi at %d th in sampled points in frame %d \n",j ,i ,frameCount);
								// 	continue;//若遇到nan跳过这个点p，也不把这里的wi,Ixi计算入内
								// }
								
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
							
							// if(std::isnan(y_i.x()))//
							// {
							// 	std::cout <<"2NaN Warning!"<<endl;
							// 	// std::cout << "xi: " << x_i << '\n';
							// 	printf("I_xi: %f \n", I_xi);//
							// 	// std::cout << "nj: " << nj << '\n';//就是因为nj是nan
							// 	// std::cout << "yi: " << y_i << '\n';//若I是nan，则yi都也是nan
							// 	printf("2NaN occur at %d th sampled points in frame %d \n",i ,frameCount);
							// 	printf("Skip this residua of xi \n");
							// 	continue;//不把该点的残差计入总loss
								
							// }
							
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

				// /*
				TicToc t_pub;
				/*
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
				*/
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

				//整个mapping的用时
				printf("whole mapping time %f ms **************************\n \n", t_whole.toc());

				nav_msgs::Odometry odomAftMapped;
				odomAftMapped.header.frame_id = "/camera_init";
				odomAftMapped.child_frame_id = "/aft_mapped";
				odomAftMapped.header.stamp = ros::Time().fromSec(timeLaserOdometry);
				odomAftMapped.pose.pose.orientation.x = q_w_curr.x();
				odomAftMapped.pose.pose.orientation.y = q_w_curr.y();
				odomAftMapped.pose.pose.orientation.z = q_w_curr.z();
				odomAftMapped.pose.pose.orientation.w = q_w_curr.w();
				odomAftMapped.pose.pose.position.x = t_w_curr(0);
				odomAftMapped.pose.pose.position.y = t_w_curr(1);
				odomAftMapped.pose.pose.position.z = t_w_curr(2);
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
				transform.setOrigin(tf::Vector3(t_w_curr(0),
												t_w_curr(1),
												t_w_curr(2)));
				q.setW(q_w_curr.w());
				q.setX(q_w_curr.x());
				q.setY(q_w_curr.y());
				q.setZ(q_w_curr.z());
				transform.setRotation(q);
				br.sendTransform(tf::StampedTransform(transform, odomAftMapped.header.stamp, "/camera_init", "/aft_mapped"));
				// */

				frameCount++;
			}
	
		}
		else //注意有些点云为空，对应的时间戳不能使用  后半程回漂 找不到特征点
		{
			
			if(!systemInited)
			{
				printf("laserMap: Initialize Done after use first %d frames odom_pose ! \n", systemDelay_map);
				systemInited = true;
			}
			
			//之后的将依靠前两帧最后mapped位姿进行插值得到初始位姿估计，而不是靠odometry的输出
			while ( !fullResBuf.empty()  && !processedBuf.empty() && !GroundLastBuf.empty() )
			{
				//线性估计的位姿若漂移，为true；否则false 
				bool driftflag = false;
				mBuf.lock();
				//确保各容器数据的时间戳是合理的
				
				// while (!odometryBuf.empty() && odometryBuf.front()->header.stamp.toSec() < fullResBuf.front()->header.stamp.toSec())
				// 	odometryBuf.pop();
				// if (odometryBuf.empty())
				// {
				// 	mBuf.unlock();
				// 	break;
				// }
				/*
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
				*/
				//待采样的cloud
				while (!processedBuf.empty() && processedBuf.front()->header.stamp.toSec() < fullResBuf.front()->header.stamp.toSec())
					processedBuf.pop();
				if (processedBuf.empty())
				{
					mBuf.unlock();
					break;//无数据，重新接收，并执行判断是否接收到
				}

				while (!GroundLastBuf.empty() && GroundLastBuf.front()->header.stamp.toSec() < fullResBuf.front()->header.stamp.toSec())
					GroundLastBuf.pop();
				if (GroundLastBuf.empty())
				{
					mBuf.unlock();
					break;//无数据，重新接收，并执行判断是否接收到
				}

				//得到时间戳
				timecloudprocessed = processedBuf.front()->header.stamp.toSec();
				timecloudGround = GroundLastBuf.front()->header.stamp.toSec();
				// timeLaserCloudCornerLast = cornerLastBuf.front()->header.stamp.toSec();
				// timeLaserCloudSurfLast = surfLastBuf.front()->header.stamp.toSec();
				timeLaserCloudFullRes = fullResBuf.front()->header.stamp.toSec();
				// timeLaserOdometry = odometryBuf.front()->header.stamp.toSec();

				//确保各类点云的时间戳同步（那前面为啥也判断了时间？）
				// if (timeLaserCloudCornerLast != timeLaserOdometry ||
				// 	timeLaserCloudSurfLast != timeLaserOdometry ||
				// 	timeLaserCloudFullRes != timeLaserOdometry || 
				// 	timecloudprocessed != timeLaserOdometry)
				if (timecloudprocessed != timeLaserCloudFullRes ||
					timecloudGround != timeLaserCloudFullRes/* ||
					timeLaserOdometry != timeLaserCloudFullRes*/)
				{
					printf("time full %f processed %f Ground %f \n", timeLaserCloudFullRes, timecloudprocessed, timecloudGround);
					printf("unsync messeage!");
					mBuf.unlock();
					break;
				}

				//数据从容器到点云指针
				// laserCloudCornerLast->clear();
				// pcl::fromROSMsg(*cornerLastBuf.front(), *laserCloudCornerLast);
				// cornerLastBuf.pop();

				// laserCloudSurfLast->clear();
				// pcl::fromROSMsg(*surfLastBuf.front(), *laserCloudSurfLast);
				// surfLastBuf.pop();

				laserCloudFullRes->clear();
				pcl::fromROSMsg(*fullResBuf.front(), *laserCloudFullRes);
				fullResBuf.pop();

				Cloudprocessed->clear();
				pcl::fromROSMsg(*processedBuf.front(), *Cloudprocessed);
				processedBuf.pop();
				
				CloudGroundCurr->clear();
				pcl::fromROSMsg(*GroundLastBuf.front(), *CloudGroundCurr);
				GroundLastBuf.pop();
				//当前帧相对于odom world 的位姿  第0帧就是单位矩阵
				//仍然需要输入进行备用！
				// q_odom_b.x() = odometryBuf.front()->pose.pose.orientation.x;
				// q_odom_b.y() = odometryBuf.front()->pose.pose.orientation.y;
				// q_odom_b.z() = odometryBuf.front()->pose.pose.orientation.z;
				// q_odom_b.w() = odometryBuf.front()->pose.pose.orientation.w;
				// t_odom_b.x() = odometryBuf.front()->pose.pose.position.x;
				// t_odom_b.y() = odometryBuf.front()->pose.pose.position.y;
				// t_odom_b.z() = odometryBuf.front()->pose.pose.position.z;
				// odometryBuf.pop();

				// 位姿的初始估计值 需要知道前两帧的优化后的最终位姿
				if(frameCount < 2 )
				{//第0帧，第1帧 位姿初始值都为I
					//第0帧 直接输出单位阵 q_wodom_curr初始值就是I
				
				}
				else//从第2帧开始，前面两帧供参考
				{	
					//poses[frameCount-1]是容器里的末尾值，是当前帧的上一帧
					q_w_k_1.x() = laserAfterMappedPath.poses[frameCount-1].pose.orientation.x;
					q_w_k_1.y() = laserAfterMappedPath.poses[frameCount-1].pose.orientation.y;
					q_w_k_1.z() = laserAfterMappedPath.poses[frameCount-1].pose.orientation.z;
					q_w_k_1.w() = laserAfterMappedPath.poses[frameCount-1].pose.orientation.w;
					t_w_k_1.x() = laserAfterMappedPath.poses[frameCount-1].pose.position.x;
					t_w_k_1.y() = laserAfterMappedPath.poses[frameCount-1].pose.position.y;
					t_w_k_1.z() = laserAfterMappedPath.poses[frameCount-1].pose.position.z;
				
					q_w_k_2.x() = laserAfterMappedPath.poses[frameCount-2].pose.orientation.x;
					q_w_k_2.y() = laserAfterMappedPath.poses[frameCount-2].pose.orientation.y;
					q_w_k_2.z() = laserAfterMappedPath.poses[frameCount-2].pose.orientation.z;
					q_w_k_2.w() = laserAfterMappedPath.poses[frameCount-2].pose.orientation.w;
					t_w_k_2.x() = laserAfterMappedPath.poses[frameCount-2].pose.position.x;
					t_w_k_2.y() = laserAfterMappedPath.poses[frameCount-2].pose.position.y;
					t_w_k_2.z() = laserAfterMappedPath.poses[frameCount-2].pose.position.z;

					//debug-验证是否是之前的位姿
					std::cout<<"the result pose of frame"<<frameCount-2<<": q= "<<q_w_k_2.coeffs().transpose()<<"\nt= "<<t_w_k_2.transpose()<<"\n"<<endl;
					std::cout<<"the result pose of frame"<<frameCount-1<<": q= "<<q_w_k_1.coeffs().transpose()<<"\nt= "<<t_w_k_1.transpose()<<"\n"<<endl;
					
					//先转换为wodom_curr
					
					// q_wodom_k_1 = q_wmap_wodom.inverse() * q_w_k_1;
					q_wodom_k_1 = q_w_k_1;
					// t_wodom_k_1 = q_wmap_wodom.inverse() * (t_w_k_1 - t_wmap_wodom);
					t_wodom_k_1 = t_w_k_1;

					// q_wodom_k_2 = q_wmap_wodom.inverse() * q_w_k_2;
					q_wodom_k_2 = q_w_k_2;
					// t_wodom_k_2 = q_wmap_wodom.inverse() * (t_w_k_2 - t_wmap_wodom);
					t_wodom_k_2 = t_w_k_2;

					//各自构成转换矩阵(4,4)
					Eigen::Isometry3d T_wodom_k_1 = Eigen::Isometry3d::Identity();
					Eigen::Isometry3d T_wodom_k_2 = Eigen::Isometry3d::Identity();
					Eigen::AngleAxisd rotation_vector_wodom_k_1(q_wodom_k_1);
					Eigen::AngleAxisd rotation_vector_wodom_k_2(q_wodom_k_2);
					T_wodom_k_1.rotate(rotation_vector_wodom_k_1);
					T_wodom_k_1.pretranslate(t_wodom_k_1);
					T_wodom_k_2.rotate(rotation_vector_wodom_k_2);
					T_wodom_k_2.pretranslate(t_wodom_k_2);

					//假设连续帧之间的运动是相似的，所以有：
					Eigen::Isometry3d T_wodom_curr = T_wodom_k_1 * T_wodom_k_2.inverse() * T_wodom_k_1;

					//获取估计的当前帧的旋转和平移
					q_wodom_curr = Eigen::Quaterniond(T_wodom_curr.matrix().topLeftCorner<3,3>());
					t_wodom_curr = T_wodom_curr.matrix().topRightCorner<3, 1>();

					std::cout<<"the init 'odom' pose of frame"<<frameCount<<": q= "<<q_wodom_curr.coeffs().transpose()<<"\nt= "<<t_wodom_curr.transpose()<<"\n"<<endl;

					//try:?另一种选择 就拿上一帧的位姿作为当前位姿初始估计？
					// 若漂移的情况 的另一种备用
					// q_odom_b = q_wodom_k_1;
					// t_odom_b = t_wodom_k_1;
					
				}
				
				// while(!cornerLastBuf.empty())
				// {
				// 	cornerLastBuf.pop();//清空该容器和实时性有什么关系呢  是否因为这个才会有跳帧？ 注释后 不在跳帧了！
				// 	printf("drop lidar frame in mapping for real time performance \n");//那为啥不对其他容器再次pop一下呢
				// }

				mBuf.unlock();

				TicToc t_whole;

				//把线性变换后的位姿转为相对于map世界坐标系的位姿 （初始时实际上乘的是单位矩阵）
				// transformAssociateToMap();//第一帧的话 应该有q_w_curr=1 0 0 0
				q_w_curr = q_wodom_curr;
				t_w_curr = t_wodom_curr;

				//********IMLS-SLAM SCAN SAMPLING STRATEGY 扫描点的采样  to do 增加地面点 没用
				TicToc t_scansample;//匹配前点云采样计时
				/*
				//先计算每个点的法线pcl::PointXYZI PointType
				pcl::NormalEstimationOMP<pcl::PointXYZI, pcl::Normal> ne;
				ne.setInputCloud (Cloudprocessed);

				pcl::search::KdTree<pcl::PointXYZI>::Ptr netree (new pcl::search::KdTree<pcl::PointXYZI>());
				ne.setSearchMethod (netree);

				pcl::PointCloud<pcl::Normal>::Ptr cloud_normals (new pcl::PointCloud<pcl::Normal>);
				//设置 半径内搜索临近点
				ne.setRadiusSearch (0.5);
				ne.compute (*cloud_normals);//得到当前scan：Cloudprocessed中每个点的法线
				*/
				//但是还要得到法线计算过程中的中间值 PCA的特征值
				kdtreeProcessed->setInputCloud(Cloudprocessed);
				int numprocessed = Cloudprocessed->points.size();
				double a_2d[numprocessed];//保存每个点的planar scalar
				
				//遍历每点计算a2d以及9个特征 and 法线
				for (int i=0; i<numprocessed; i++)
				{
					pointSel = Cloudprocessed->points[i];
					//对于当前点在scan中搜索指定半径内的近邻
					kdtreeProcessed->radiusSearch(pointSel, SampNeiThr, pointSearchInd, pointSearchSqDis);
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
					//注意这里还有一个正则因子！ 修正！
					covMat = covMat * (1/double(numneighbor));
					Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> saes(covMat);//协方差矩阵特征值分解
					// note Eigen library sort eigenvalues in increasing order
					//得到三个特征值 从大到小
					double lamda_1 = sqrt(saes.eigenvalues()[2]);
					double lamda_2 = sqrt(saes.eigenvalues()[1]);
					double lamda_3 = sqrt(saes.eigenvalues()[0]);

					//获取最小特征值对应的特征向量 即为法线
					Eigen::Vector3d rawnormcurr = saes.eigenvectors().col(0);;

					//计算当前点近邻的planar scale
					a_2d[i] = (lamda_2 - lamda_3)/lamda_1;

					Eigen::Vector3d X_axis(1, 0, 0);
					Eigen::Vector3d Y_axis(0, 1, 0);
					Eigen::Vector3d Z_axis(0, 0, 1);

					Eigen::Vector3d pointcurr(Cloudprocessed->points[i].x,
											Cloudprocessed->points[i].y,
											Cloudprocessed->points[i].z);
					//该点的法线
					// Eigen::Vector3d pointnormcurr(cloud_normals->points[i].normal_x,
					// 						cloud_normals->points[i].normal_y,
					// 						cloud_normals->points[i].normal_z);

					Eigen::Vector3d pointnormcurr = rawnormcurr.normalized();//归一化
					
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

				laserCloudFromMap->clear();
				//从地图数组中得到当前所有modle point
				for (int i = 0; i < mapsize; i++)//100
				{
					// pcl::PointCloud<PointType>::Ptr ModelPointCloud[i](new pcl::PointCloud<PointType>());//要初始化？to debug
					// pcl::PointCloud<PointType> ModelPointCloud[i];//bug 不用指针数组后 就ok了 不再需要这样的初始化

					*laserCloudFromMap += ModelPointCloud[i];//
				}
				int laserCloudMapNum = laserCloudFromMap->points.size();
			
				//清空之前存储的先前帧的采样点
				//CloudSampled->clear();
				CloudSampled.reset(new pcl::PointCloud<PointType>());
				if(CloudSampled->points.size() != 0)
				{
					std::cout<<"Failed to clear CloudSampled !\n"<<endl;
				}
				/*
				//试一试：把当前帧的地面点也作为采样特征点去进行匹配
				//先对地面点进行下采样吧，否则点太多了
				pcl::PointCloud<PointType>::Ptr CloudGroundDS(new pcl::PointCloud<PointType>());//下采样后的地面特征点
				downSizeFilterSurf.setInputCloud(CloudGroundCurr);
				downSizeFilterSurf.filter(*CloudGroundDS);
				int numGroundDS = CloudGroundDS->points.size();
				*/
				if(laserCloudMapNum==0)//当前map model是空的，就直接取9个列表的较大的s个点
				{
					std::cout<<"current map model is empty !"<<endl;
					/*
					for (int i = 0; i < numGroundDS; i++)//选取所有地面点
					{
						
						CloudSampled->push_back(CloudGroundDS->points[i]);
					}
					*/

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
					printf("points size of current map model: %d before mapping Frame %d \n", laserCloudMapNum, frameCount);//输出当前地图的大小

					TicToc t_tree;
					//建立model points kd tree
					kdtreeFromMap->setInputCloud(laserCloudFromMap);
					printf("build tree of map time %f ms \n", t_tree.toc());//建立地图点kdtree的时间
					/*
					for (int i = 0; i < numGroundDS; i++)//遍历所有地面点 ,只要非野点都要
					{
						pointOri = CloudGroundDS->points[i];
						pointAssociateToMap(&pointOri, &pointSel);//将该采样点转为map世界坐标系  使用的是odom的结果位姿 这步转换是否需要？
						//在现有地图点中找到距离采样点最近的一个点
						kdtreeFromMap->nearestKSearch(pointSel, 1, pointSearchInd, pointSearchSqDis);
						if (sqrt(pointSearchSqDis[0]) > SampRadThr)//此两点之间距离太大，认为是野点 
						{
							continue;//跳过，下个点
						}
						
						CloudSampled->push_back(CloudGroundDS->points[i]);//注意加入的是变换前的点！
					}
					*/
					int numpicked = 0;//计数
					for (int i = 0; i < numprocessed; i++)//遍历所有点 直到找满100个点
					{
						int ind1 = cloudSortInd1[i];//值较大点的索引
						pointOri = Cloudprocessed->points[ind1];
						pointAssociateToMap(&pointOri, &pointSel);//将该采样点转为map世界坐标系  使用的是odom的结果位姿 这步转换是否需要？
						//在现有地图点中找到距离采样点最近的一个点
						kdtreeFromMap->nearestKSearch(pointSel, 1, pointSearchInd, pointSearchSqDis);
						if (sqrt(pointSearchSqDis[0]) > SampRadThr)//此两点之间距离太大，认为是野点 
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
					// printf("select %d points in list 1 \n", numpicked);

					numpicked = 0;//计数器清零
					for (int i = 0; i < numprocessed; i++)//遍历所有点 直到找满100个点
					{
						int ind2 = cloudSortInd2[i];//值较大点的索引
						pointOri = Cloudprocessed->points[ind2];
						pointAssociateToMap(&pointOri, &pointSel);//将该采样点转为map世界坐标系  使用的是odom的结果位姿
						//在现有地图点中找到距离采样点最近的一个点
						kdtreeFromMap->nearestKSearch(pointSel, 1, pointSearchInd, pointSearchSqDis);
						if (sqrt(pointSearchSqDis[0]) > SampRadThr)//此两点之间距离太大，认为是野点 
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
					// printf("select %d points in list 2 \n", numpicked);
					numpicked = 0;//计数器清零
					for (int i = 0; i < numprocessed; i++)//遍历所有点 直到找满100个点
					{
						int ind3 = cloudSortInd3[i];//值较大点的索引
						pointOri = Cloudprocessed->points[ind3];
						pointAssociateToMap(&pointOri, &pointSel);//将该采样点转为map世界坐标系  使用的是odom的结果位姿
						//在现有地图点中找到距离采样点最近的一个点
						kdtreeFromMap->nearestKSearch(pointSel, 1, pointSearchInd, pointSearchSqDis);
						if (sqrt(pointSearchSqDis[0]) > SampRadThr)//此两点之间距离太大，认为是野点 
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
					// printf("select %d points in list 3 \n", numpicked);
					numpicked = 0;//计数器清零
					for (int i = 0; i < numprocessed; i++)//遍历所有点 直到找满100个点
					{
						int ind4 = cloudSortInd4[i];//值较大点的索引
						pointOri = Cloudprocessed->points[ind4];
						pointAssociateToMap(&pointOri, &pointSel);//将该采样点转为map世界坐标系  使用的是odom的结果位姿
						//在现有地图点中找到距离采样点最近的一个点
						kdtreeFromMap->nearestKSearch(pointSel, 1, pointSearchInd, pointSearchSqDis);
						if (sqrt(pointSearchSqDis[0]) > SampRadThr)//此两点之间距离太大，认为是野点 
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
					// printf("select %d points in list 4 \n", numpicked);
					numpicked = 0;//计数器清零
					for (int i = 0; i < numprocessed; i++)//遍历所有点 直到找满100个点
					{
						int ind5 = cloudSortInd5[i];//值较大点的索引
						pointOri = Cloudprocessed->points[ind5];
						pointAssociateToMap(&pointOri, &pointSel);//将该采样点转为map世界坐标系  使用的是odom的结果位姿
						//在现有地图点中找到距离采样点最近的一个点
						kdtreeFromMap->nearestKSearch(pointSel, 1, pointSearchInd, pointSearchSqDis);
						if (sqrt(pointSearchSqDis[0]) > SampRadThr)//此两点之间距离太大，认为是野点 
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
					// printf("select %d points in list 5 \n", numpicked);
					numpicked = 0;//计数器清零
					for (int i = 0; i < numprocessed; i++)//遍历所有点 直到找满100个点
					{
						int ind6 = cloudSortInd6[i];//值较大点的索引
						pointOri = Cloudprocessed->points[ind6];
						pointAssociateToMap(&pointOri, &pointSel);//将该采样点转为map世界坐标系  使用的是odom的结果位姿
						//在现有地图点中找到距离采样点最近的一个点
						kdtreeFromMap->nearestKSearch(pointSel, 1, pointSearchInd, pointSearchSqDis);
						if (sqrt(pointSearchSqDis[0]) > SampRadThr)//此两点之间距离太大，认为是野点 
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
					// printf("select %d points in list 6 \n", numpicked);
					numpicked = 0;//计数器清零
					for (int i = 0; i < numprocessed; i++)//遍历所有点 直到找满100个点
					{
						int ind7 = cloudSortInd7[i];//值较大点的索引
						pointOri = Cloudprocessed->points[ind7];
						pointAssociateToMap(&pointOri, &pointSel);//将该采样点转为map世界坐标系  使用的是odom的结果位姿
						//在现有地图点中找到距离采样点最近的一个点
						kdtreeFromMap->nearestKSearch(pointSel, 1, pointSearchInd, pointSearchSqDis);
						if (sqrt(pointSearchSqDis[0]) > SampRadThr)//此两点之间距离太大，认为是野点 
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
					// printf("select %d points in list 7 \n", numpicked);
					numpicked = 0;//计数器清零
					for (int i = 0; i < numprocessed; i++)//遍历所有点 直到找满100个点
					{
						int ind8 = cloudSortInd8[i];//值较大点的索引
						pointOri = Cloudprocessed->points[ind8];
						pointAssociateToMap(&pointOri, &pointSel);//将该采样点转为map世界坐标系  使用的是odom的结果位姿
						//在现有地图点中找到距离采样点最近的一个点
						kdtreeFromMap->nearestKSearch(pointSel, 1, pointSearchInd, pointSearchSqDis);
						if (sqrt(pointSearchSqDis[0]) > SampRadThr)//此两点之间距离太大，认为是野点 
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
					// printf("select %d points in list 8 \n", numpicked);
					numpicked = 0;//计数器清零
					for (int i = 0; i < numprocessed; i++)//遍历所有点 直到找满100个点
					{
						int ind9 = cloudSortInd9[i];//值较大点的索引
						pointOri = Cloudprocessed->points[ind9];
						pointAssociateToMap(&pointOri, &pointSel);//将该采样点转为map世界坐标系  使用的是odom的结果位姿
						//在现有地图点中找到距离采样点最近的一个点
						kdtreeFromMap->nearestKSearch(pointSel, 1, pointSearchInd, pointSearchSqDis);//这里是平方距离！
						if (sqrt(pointSearchSqDis[0]) > SampRadThr)//此两点之间距离太大，认为是野点 
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
					// printf("select %d points in list 9 \n", numpicked);

				}
				
				//这是意味着当前位姿初始估计值太偏了 导致成为 outlier
				// if(CloudSampled->points.size()==0)
				//if(CloudSampled->points.size() < 900) //to 改进 ！·
				/*
				while(CloudSampled->points.size() < 900)
				{
					printf("WARNING: the frame %d has %d < 900 sampled points ! \n", frameCount, int(CloudSampled->points.size()) );
					driftflag = true; //标记当前帧要使用odo重新采样
					//需要重新生成位姿初始值来重新进行变换 采样特征点 在q_w_curr插值 随机地
					double s;
					srand((unsigned)time(NULL));
					s = rand() / double(RAND_MAX); //(0,1)之间随机数
					printf("random s: %f \n", s);
					//插值
					q_odom_b = q_wodom_k_2.slerp(s, q_wodom_curr);
					t_odom_b = t_wodom_k_2 + s * (t_wodom_curr - t_wodom_k_2);

					transformAssociateToMap_b();//这里使用odo结果得到新的T_w_curr  
					CloudSampled.reset(new pcl::PointCloud<PointType>());//先清空前面不太对的采样点
					if(CloudSampled->points.size() != 0)
					{
						std::cout<<"Failed to clear CloudSampled !\n"<<endl;
					}
					int numpicked = 0;//计数
					for (int i = 0; i < numprocessed; i++)//遍历所有点 直到找满100个点
					{
						int ind1 = cloudSortInd1[i];//值较大点的索引
						pointOri = Cloudprocessed->points[ind1];
						pointAssociateToMap(&pointOri, &pointSel);//将该采样点转为map世界坐标系  使用的是odom的结果位姿
						//在现有地图点中找到距离采样点最近的一个点
						kdtreeFromMap->nearestKSearch(pointSel, 1, pointSearchInd, pointSearchSqDis);
						if (sqrt(pointSearchSqDis[0]) > SampRadThr)//此两点之间距离太大，认为是野点 
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
					printf("again select %d points in list 1 \n", numpicked);

					numpicked = 0;//计数器清零
					for (int i = 0; i < numprocessed; i++)//遍历所有点 直到找满100个点
					{
						int ind2 = cloudSortInd2[i];//值较大点的索引
						pointOri = Cloudprocessed->points[ind2];
						pointAssociateToMap(&pointOri, &pointSel);//将该采样点转为map世界坐标系  使用的是odom的结果位姿
						//在现有地图点中找到距离采样点最近的一个点
						kdtreeFromMap->nearestKSearch(pointSel, 1, pointSearchInd, pointSearchSqDis);
						if (sqrt(pointSearchSqDis[0]) > SampRadThr)//此两点之间距离太大，认为是野点 
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
					printf("again select %d points in list 2 \n", numpicked);
					numpicked = 0;//计数器清零
					for (int i = 0; i < numprocessed; i++)//遍历所有点 直到找满100个点
					{
						int ind3 = cloudSortInd3[i];//值较大点的索引
						pointOri = Cloudprocessed->points[ind3];
						pointAssociateToMap(&pointOri, &pointSel);//将该采样点转为map世界坐标系  使用的是odom的结果位姿
						//在现有地图点中找到距离采样点最近的一个点
						kdtreeFromMap->nearestKSearch(pointSel, 1, pointSearchInd, pointSearchSqDis);
						if (sqrt(pointSearchSqDis[0]) > SampRadThr)//此两点之间距离太大，认为是野点 
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
					printf("again select %d points in list 3 \n", numpicked);
					numpicked = 0;//计数器清零
					for (int i = 0; i < numprocessed; i++)//遍历所有点 直到找满100个点
					{
						int ind4 = cloudSortInd4[i];//值较大点的索引
						pointOri = Cloudprocessed->points[ind4];
						pointAssociateToMap(&pointOri, &pointSel);//将该采样点转为map世界坐标系  使用的是odom的结果位姿
						//在现有地图点中找到距离采样点最近的一个点
						kdtreeFromMap->nearestKSearch(pointSel, 1, pointSearchInd, pointSearchSqDis);
						if (sqrt(pointSearchSqDis[0]) > SampRadThr)//此两点之间距离太大，认为是野点 
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
					printf("again select %d points in list 4 \n", numpicked);
					numpicked = 0;//计数器清零
					for (int i = 0; i < numprocessed; i++)//遍历所有点 直到找满100个点
					{
						int ind5 = cloudSortInd5[i];//值较大点的索引
						pointOri = Cloudprocessed->points[ind5];
						pointAssociateToMap(&pointOri, &pointSel);//将该采样点转为map世界坐标系  使用的是odom的结果位姿
						//在现有地图点中找到距离采样点最近的一个点
						kdtreeFromMap->nearestKSearch(pointSel, 1, pointSearchInd, pointSearchSqDis);
						if (sqrt(pointSearchSqDis[0]) > SampRadThr)//此两点之间距离太大，认为是野点 
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
					printf("again select %d points in list 5 \n", numpicked);
					numpicked = 0;//计数器清零
					for (int i = 0; i < numprocessed; i++)//遍历所有点 直到找满100个点
					{
						int ind6 = cloudSortInd6[i];//值较大点的索引
						pointOri = Cloudprocessed->points[ind6];
						pointAssociateToMap(&pointOri, &pointSel);//将该采样点转为map世界坐标系  使用的是odom的结果位姿
						//在现有地图点中找到距离采样点最近的一个点
						kdtreeFromMap->nearestKSearch(pointSel, 1, pointSearchInd, pointSearchSqDis);
						if (sqrt(pointSearchSqDis[0]) > SampRadThr)//此两点之间距离太大，认为是野点 
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
					printf("again select %d points in list 6 \n", numpicked);
					numpicked = 0;//计数器清零
					for (int i = 0; i < numprocessed; i++)//遍历所有点 直到找满100个点
					{
						int ind7 = cloudSortInd7[i];//值较大点的索引
						pointOri = Cloudprocessed->points[ind7];
						pointAssociateToMap(&pointOri, &pointSel);//将该采样点转为map世界坐标系  使用的是odom的结果位姿
						//在现有地图点中找到距离采样点最近的一个点
						kdtreeFromMap->nearestKSearch(pointSel, 1, pointSearchInd, pointSearchSqDis);
						if (sqrt(pointSearchSqDis[0]) > SampRadThr)//此两点之间距离太大，认为是野点 
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
					printf("again select %d points in list 7 \n", numpicked);
					numpicked = 0;//计数器清零
					for (int i = 0; i < numprocessed; i++)//遍历所有点 直到找满100个点
					{
						int ind8 = cloudSortInd8[i];//值较大点的索引
						pointOri = Cloudprocessed->points[ind8];
						pointAssociateToMap(&pointOri, &pointSel);//将该采样点转为map世界坐标系  使用的是odom的结果位姿
						//在现有地图点中找到距离采样点最近的一个点
						kdtreeFromMap->nearestKSearch(pointSel, 1, pointSearchInd, pointSearchSqDis);
						if (sqrt(pointSearchSqDis[0]) > SampRadThr)//此两点之间距离太大，认为是野点 
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
					printf("again select %d points in list 8 \n", numpicked);
					numpicked = 0;//计数器清零
					for (int i = 0; i < numprocessed; i++)//遍历所有点 直到找满100个点
					{
						int ind9 = cloudSortInd9[i];//值较大点的索引
						pointOri = Cloudprocessed->points[ind9];
						pointAssociateToMap(&pointOri, &pointSel);//将该采样点转为map世界坐标系  使用的是odom的结果位姿
						//在现有地图点中找到距离采样点最近的一个点
						kdtreeFromMap->nearestKSearch(pointSel, 1, pointSearchInd, pointSearchSqDis);//这里是平方距离！
						if (sqrt(pointSearchSqDis[0]) > SampRadThr)//此两点之间距离太大，认为是野点 
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
					printf("again select %d points in list 9 \n", numpicked);

				}
				*/
				// else
				// {
				// 	// std::cout << "the size of sampled points of frame " << frameCount << " is " << CloudSampled->points.size() << '\n';
				// }
				printf("scan sampling time %f ms \n", t_scansample.toc());//采样总时间
				int numscansamped = CloudSampled->points.size();//使用估计的位姿，会出现采样点为0的情况
				//采样前后的点数变化  正常是采到900点 多出的就是地面点  (with dsground)
				std::cout << "the size before sampled : " << numprocessed << " and the size after sampled is " << numscansamped << '\n';
				
				//发布采样前的点
				sensor_msgs::PointCloud2 CloudbeforeSampled;//
				pcl::toROSMsg(*Cloudprocessed, CloudbeforeSampled);
				CloudbeforeSampled.header.stamp = ros::Time().fromSec(timeLaserCloudFullRes);
				CloudbeforeSampled.header.frame_id = "/camera_init";
				pubCloudProcessed.publish(CloudbeforeSampled);
				//for now 发布采样后的特征点
				sensor_msgs::PointCloud2 laserCloudSampled;//
				pcl::toROSMsg(*CloudSampled, laserCloudSampled);
				laserCloudSampled.header.stamp = ros::Time().fromSec(timeLaserCloudFullRes);
				laserCloudSampled.header.frame_id = "/camera_init";
				pubCloudSampled.publish(laserCloudSampled);

				//地图中的特征点数目满足要求  若是第0帧 不执行这部分 包含匹配以及优化 to improve!
				if (laserCloudMapNum > 10)
				{
					//先计算现有地图点云中的点的法线 在一次mapping中这是不变的量，用于下面求Yk
					/*
					pcl::NormalEstimationOMP<pcl::PointXYZI, pcl::Normal> nem;
					nem.setInputCloud (laserCloudFromMap);

					pcl::search::KdTree<pcl::PointXYZI>::Ptr nemtree (new pcl::search::KdTree<pcl::PointXYZI>());
					nem.setSearchMethod (nemtree);

					pcl::PointCloud<pcl::Normal>::Ptr mapcloud_normals (new pcl::PointCloud<pcl::Normal>);
					//设置 半径内搜索临近点
					// nem.setRadiusSearch (0.5); //可调
					nem.setKSearch (numkneibor);//保证一定能计算有效法线 可调5 8? 9
					nem.compute (*mapcloud_normals);//得到map中每个点的法线 会出现nan的情况 那是因为没有找到指定半径的邻域
					*/
					TicToc t_opt;
					//以下有bug出没
					//ceres优化求解 迭代20次 mapping 过程
					for (int iterCount = 0; iterCount < numICP; iterCount++)  //debug 2次
					{
						//每次迭代 都要做下面的事：

						//优化相关 好像优化的结果和这个lossfunc有关 null步子太大  CauchyLoss
						// ceres::LossFunction *loss_function = NULL;
						ceres::LossFunction *loss_function = new ceres::HuberLoss(lossarg);//0.1走到负方向了
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
						//记录每次优化被加入残差项的点的个数
						int featurepointadded = 0;
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
							// Eigen::Vector3d nj(mapcloud_normals->points[ pointSearchInd[0] ].normal_x,
							// 								mapcloud_normals->points[ pointSearchInd[0] ].normal_y,
							// 								mapcloud_normals->points[ pointSearchInd[0] ].normal_z);

							//最近点的法线n_j 论文的意思是只需计算最近点法线，其他box近邻内的法线值去一样的值近似！
							PointType nearestp = laserCloudFromMap->points[ pointSearchInd[0] ];
							Eigen::Vector3d nearestv(laserCloudFromMap->points[ pointSearchInd[0] ].x,
													laserCloudFromMap->points[ pointSearchInd[0] ].y,
													laserCloudFromMap->points[ pointSearchInd[0] ].z);
							
							//寻找地图点中关于nearestp的邻域 拟合平面 计算法线
							kdtreeFromMap->radiusSearch(nearestp, SampNeiThr, pointNeighborInd, pointNeighborSqDis);
							int numneighborm = pointNeighborInd.size();//得到的半径内近邻个数
							std::vector<Eigen::Vector3d> neighborsm;//存储若干近邻点
							Eigen::Vector3d centerm(0, 0, 0);//初始化近邻点的重心
							for (int j = 0; j < numneighborm; j++)
							{
								Eigen::Vector3d tmpm(laserCloudFromMap->points[pointNeighborInd[j]].x,
													laserCloudFromMap->points[pointNeighborInd[j]].y,
													laserCloudFromMap->points[pointNeighborInd[j]].z);
								centerm = centerm + tmpm;
								neighborsm.push_back(tmpm);
							}
							//得到近邻点坐标的重心
							centerm = centerm / double(numneighborm);

							Eigen::Matrix3d covMatm = Eigen::Matrix3d::Zero();//近邻点的协方差矩阵
							for (int j = 0; j < numneighborm; j++)
							{
								Eigen::Matrix<double, 3, 1> tmpZeroMeanm = neighborsm[j] - centerm;
								covMatm = covMatm + tmpZeroMeanm * tmpZeroMeanm.transpose();
							}
							//注意这里还有一个正则因子！ 修正！
							covMatm = covMatm * (1/double(numneighborm));
							Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> saesm(covMatm);//协方差矩阵特征值分解
							// note Eigen library sort eigenvalues in increasing order

							//获取最小特征值对应的特征向量 即为法线
							Eigen::Vector3d rawnorm = saesm.eigenvectors().col(0);


							if(std::isnan(rawnorm.x()))//
							{
								std::cout <<"nj NaN Warning!"<<endl;
								// std::cout << "nj: " << rawnorm << '\n';//就是因为nj是nan
								printf("nj NaN occur at %d th sampled points in frame %d @ %d iteration\n",i ,frameCount, iterCount);
								printf("Skip this residua of xi \n");
								continue;//不把该点的残差计入总loss	
							}

							Eigen::Vector3d nj = rawnorm.normalized();//归一化
							if( nearestv.dot(nj) > 0)
							{//法线方向统一
								nj = -nj;
							}

							for (int j = 0; j < numBox; j++)
							{	//当前来自地图中的点p_
								Eigen::Vector3d pcurrbox(laserCloudFromMap->points[ pointSearchInd[j] ].x,
														laserCloudFromMap->points[ pointSearchInd[j] ].y,
														laserCloudFromMap->points[ pointSearchInd[j] ].z);
								//当前来自地图中的点p_的法线
								// Eigen::Vector3d normpcurr(mapcloud_normals->points[ pointSearchInd[j] ].normal_x,
								// 							mapcloud_normals->points[ pointSearchInd[j] ].normal_y,
								// 							mapcloud_normals->points[ pointSearchInd[j] ].normal_z);
								Eigen::Vector3d normpcurr = nj;
								//当前点对应的权值
								double w_j = exp(-pointSearchSqDis[j]/(h_imls*h_imls));
								/*
								//to debug
								if (std::isnan(normpcurr.x()))//tmp1
								{
									std::cout <<"iNaN Warning!"<<endl;//还是会出现
									
									printf("iNaN the %d item of vector Ixi at %d th sampled points in frame %d \n",j ,i ,frameCount);
									continue;//若遇到nan跳过这个点p，也不把这里的wi,Ixi计算入内
								}
								*/
								double tmp1 = w_j*((pointx-pcurrbox).dot(normpcurr));//某一项会有nan？
								Wx.push_back(w_j);
								Ixi.push_back(tmp1);
							}
							// printf("bug876 \n");//以上都ok
							//计算采样点x到map点隐式平面的距离
							double fenzi = std::accumulate(Ixi.begin(), Ixi.end(), 0.000001);//出现了负值？合理吗
							// printf("fenzi %f \n", fenzi);//分子首先有nan！
							double fenmu = std::accumulate(Wx.begin(), Wx.end(), 0.000001);

							double I_xi = fenzi/fenmu;//会出现NaN
							// printf("I_xi %f \n", I_xi);
							// float I_xi = std::accumulate(Ix1.begin(), Ix1.end(), 0)/std::accumulate(Wx.begin(), Wx.end(), 0);//bug!
							// printf("bug884 \n");
							//x_i对应的点y_i
							Eigen::Vector3d y_i = pointx - I_xi * nj;
							/*
							if(std::isnan(y_i.x()))//
							{
								std::cout <<"2NaN Warning!"<<endl;
								printf("I_xi: %f \n", I_xi);//
								// std::cout << "nj: " << nj << '\n';//就是因为nj是nan
								// std::cout << "yi: " << y_i << '\n';//若I是nan，则yi都也是nan
								printf("2NaN occur at %d th sampled points in frame %d \n",i ,frameCount);
								printf("Skip this residua of xi \n");
								continue;//不把该点的残差计入总loss
								
							}
							*/
							ceres::CostFunction *cost_function = LidarPoint2PlaneICP::Create(x_i, y_i, nj);
							problem.AddResidualBlock(cost_function, loss_function, parameters, parameters + 4);
							
							featurepointadded = featurepointadded + 1;
						}
						printf("%d feature points are added to ResidualBlock @ %d th Iteration solver \n", featurepointadded, iterCount);
						// printf("bug881 \n");
						//求解优化
						TicToc t_solver;
						ceres::Solver::Options options;
						options.linear_solver_type = ceres::DENSE_QR;//属于列文伯格-马夸尔特方法
						options.max_num_iterations = 4;//一次优化的最大迭代次数
						options.minimizer_progress_to_stdout = false;//输出到cout 
						options.check_gradients = true;//开了检查梯度，发现我的优化有问题 应该是目前问题所在！
						options.gradient_check_relative_precision = 1e02;//1e-4是否太苛刻  好像是这个原因
						ceres::Solver::Summary summary;
						ceres::Solve(options, &problem, &summary);//开始优化
						// printf("the %d mapping solver time %f ms \n",iterCount , t_solver.toc());
						
						//输出一次mapping优化得到的位姿 w x y z 当前帧相对于map world的变换 
						printf("result q %f %f %f %f result t %f %f %f\n", parameters[3], parameters[0], parameters[1], parameters[2],
							   parameters[4], parameters[5], parameters[6]);
						//输出报告 debug
						std::cout<< summary.BriefReport() <<endl;
		
					}
					
					printf("\nthe frame %d mapping optimization time %f \n", frameCount, t_opt.toc());
					//20次优化后的该帧位姿最后结果 与948的值是一样的
					std::cout<<"the final result of frame"<< frameCount <<": q_w_curr= "<< q_w_curr.coeffs().transpose() <<"\nt_w_curr= "<< t_w_curr.transpose() <<"\n"<<endl;
				}
				else//点太少 一般是第0帧
				{
					ROS_WARN("Current map model points num are not enough, skip Optimization !");
				}
				
				//迭代优化结束 更新相关的转移矩阵
				// transformUpdate(); //更新了odo world 相对于 map world的变换
				/*
				if(driftflag)
				{//若临时使用odom结果
					transformUpdate_b(); //更新了odo world 相对于 map world的变换
				}
				else
				{
					transformUpdate();
				}
				*/
				TicToc t_add;
				//将当前帧的点加入到modelpoint 中 相应位置
				if (frameCount<100)//说明当前model point 还没存满 直接添加
				{
					for (int i = 0; i < numprocessed; i++)//把当前优化过的scan所有点注册到地图数组指针中
					{	//将该点转移到世界坐标系下
						pointAssociateToMap(&Cloudprocessed->points[i], &pointSel);
						ModelPointCloud[frameCount].push_back(pointSel);
					}

					if(int(ModelPointCloud[frameCount].points.size()) != numprocessed)
					{
						std::cout<<"ERROR when add point to modelpointcloud[ "<<frameCount<<" ] ! "<<endl;
						// std::cout<<"num of ModelPointCloud["<<j<<"] : "<< int(ModelPointCloud[j].points.size) <<"\n"<<endl;
					}


				}
				else//当前model point数组已填满100帧 去除第一个，从后面添加新的
				{
					for (int j = 0; j < mapsize-1; j++)
					{
						pcl::PointCloud<PointType>::Ptr tmpCloud(new pcl::PointCloud<PointType>());
						*tmpCloud = ModelPointCloud[j+1];
						int numtmpCloud = tmpCloud->points.size();
						std::cout<<"num of ModelPointCloud["<<j+1<<"] : "<< numtmpCloud <<"\n"<<endl;
						//把数组中依次前移
						ModelPointCloud[j].clear();//->
						//应该为0
						std::cout<<"num of ModelPointCloud["<<j<<"] after clear : "<< int(ModelPointCloud[j].points.size()) <<"\n"<<endl;
						// ModelPointCloud[j].reset(new pcl::PointCloud<PointType>());
						for (int k = 0; k < numtmpCloud; k++)
						{
							ModelPointCloud[j].push_back(tmpCloud->points[k]);
						}
						if(int(ModelPointCloud[j].points.size()) != numtmpCloud)
						{
							std::cout<<"ERROR when moving forward modelpointcloud! "<<endl;
							std::cout<<"num of ModelPointCloud["<<j<<"] : "<< int(ModelPointCloud[j].points.size()) <<"\n"<<endl;
						}
						
						// ModelPointCloud[j] = ModelPointCloud[j+1];
					}
					// ModelPointCloud[mapsize-1].reset(new pcl::PointCloud<PointType>());
					ModelPointCloud[mapsize-1].clear();//->
					if(int(ModelPointCloud[mapsize-1].points.size()) != 0)
					{
						std::cout<<"ERROR when clear modelpointcloud[99]! "<<endl;
						// std::cout<<"num of ModelPointCloud["<<j<<"] : "<< int(ModelPointCloud[j].points.size) <<"\n"<<endl;
					}
					//把当前帧的点注册后加入到数组最后一个位置
					for (int i = 0; i < numprocessed; i++)//把当前优化过的scan所有点注册到地图数组指针中
					{	//将该点转移到世界坐标系下
						pointAssociateToMap(&Cloudprocessed->points[i], &pointSel);
						ModelPointCloud[mapsize-1].push_back(pointSel);
					}
					if(int(ModelPointCloud[mapsize-1].points.size()) != numprocessed)
					{
						std::cout<<"ERROR when add point to modelpointcloud[99]! "<<endl;
						// std::cout<<"num of ModelPointCloud["<<j<<"] : "<< int(ModelPointCloud[j].points.size) <<"\n"<<endl;
					}
					
				}
				printf("add points time %f ms\n", t_add.toc());

				TicToc t_pub;
				/*
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
					laserCloudSurround3.header.stamp = ros::Time().fromSec(timeLaserCloudFullRes);
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
					laserCloudMsg.header.stamp = ros::Time().fromSec(timeLaserCloudFullRes);
					laserCloudMsg.header.frame_id = "/camera_init";
					pubLaserCloudMap.publish(laserCloudMsg);
				}
				*/
				//将点云中全部点(包含运动物体)转移到世界坐标系下  
				int laserCloudFullResNum = laserCloudFullRes->points.size();
				for (int i = 0; i < laserCloudFullResNum; i++)
				{
					pointAssociateToMap(&laserCloudFullRes->points[i], &laserCloudFullRes->points[i]);
				}

				sensor_msgs::PointCloud2 laserCloudFullRes3;//当前帧的所有点  多于Cloudprocessed
				pcl::toROSMsg(*laserCloudFullRes, laserCloudFullRes3);
				laserCloudFullRes3.header.stamp = ros::Time().fromSec(timeLaserCloudFullRes);
				laserCloudFullRes3.header.frame_id = "/camera_init";
				pubLaserCloudFullRes.publish(laserCloudFullRes3);///velodyne_cloud_registered 当前帧已注册的点

				printf("mapping pub time %f ms \n", t_pub.toc());

				//整个mapping的用时
				printf("whole mapping time %f ms **************************\n \n", t_whole.toc());

				// /*
				nav_msgs::Odometry odomAftMapped;
				odomAftMapped.header.frame_id = "/camera_init";
				odomAftMapped.child_frame_id = "/aft_mapped";
				odomAftMapped.header.stamp = ros::Time().fromSec(timeLaserCloudFullRes);
				odomAftMapped.pose.pose.orientation.x = q_w_curr.x();
				odomAftMapped.pose.pose.orientation.y = q_w_curr.y();
				odomAftMapped.pose.pose.orientation.z = q_w_curr.z();
				odomAftMapped.pose.pose.orientation.w = q_w_curr.w();
				odomAftMapped.pose.pose.position.x = t_w_curr(0);
				odomAftMapped.pose.pose.position.y = t_w_curr(1);
				odomAftMapped.pose.pose.position.z = t_w_curr(2);
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
				transform.setOrigin(tf::Vector3(t_w_curr(0),
												t_w_curr(1),
												t_w_curr(2)));
				q.setW(q_w_curr.w());
				q.setX(q_w_curr.x());
				q.setY(q_w_curr.y());
				q.setZ(q_w_curr.z());
				transform.setRotation(q);
				br.sendTransform(tf::StampedTransform(transform, odomAftMapped.header.stamp, "/camera_init", "/aft_mapped"));

				frameCount++;
			}

		}
		
		std::chrono::milliseconds dura(2);
        std::this_thread::sleep_for(dura);
	}
}


int main(int argc, char **argv)
{
	ros::init(argc, argv, "laserMapping");
	ros::NodeHandle nh;

	float lineRes = 0;
	float planeRes = 0;
	nh.param<float>("mapping_line_resolution", lineRes, 0.4); //aloam_velodyne_HDL_64.launch中的参数设置
	nh.param<float>("mapping_plane_resolution", planeRes, 0.8);
	printf("line resolution %f plane resolution %f \n", lineRes, planeRes);
	//设置体素栅格滤波器 体素大小
	// downSizeFilterCorner.setLeafSize(lineRes, lineRes,lineRes);
	downSizeFilterSurf.setLeafSize(planeRes, planeRes, planeRes); //用于下采样地面特征点！

	//接收来自laserodo.cpp发来的处理后的点云/cloud_4 topic

	ros::Subscriber subCloudprocessed = nh.subscribe<sensor_msgs::PointCloud2>("/cloud_4", 100, CloudprocessedHandler);

	//这3个只有前十帧才非空
	ros::Subscriber subLaserCloudCornerLast = nh.subscribe<sensor_msgs::PointCloud2>("/laser_cloud_corner_last", 100, laserCloudCornerLastHandler);

	ros::Subscriber subLaserCloudSurfLast = nh.subscribe<sensor_msgs::PointCloud2>("/laser_cloud_surf_last", 100, laserCloudSurfLastHandler);

	ros::Subscriber subLaserOdometry = nh.subscribe<nav_msgs::Odometry>("/laser_odom_to_init", 100, laserOdometryHandler);

	ros::Subscriber subLaserCloudFullRes = nh.subscribe<sensor_msgs::PointCloud2>("/velodyne_cloud_3", 100, laserCloudFullResHandler);
	//订阅odometry部分得到的当前帧的地面点
	ros::Subscriber subCloudGroundLast = nh.subscribe<sensor_msgs::PointCloud2>("/cloud_3_Ground", 100, CloudGroundLastHandler);
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