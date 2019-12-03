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
#include <fstream>
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

#include <pcl/filters/passthrough.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/approximate_voxel_grid.h>
#include <pcl/filters/radius_outlier_removal.h>
#include <pcl/filters/statistical_outlier_removal.h>

using namespace std;
using namespace Eigen;

int frameCount = 0; // 计数！
//时间戳
double timecloudprocessed = 0;
double timecloudGround = 0;
double timeLaserCloudCornerLast = 0;
double timeLaserCloudSurfLast = 0;
double timeLaserCloudFullRes = 0;
double timeLaserOdometry = 0;

//keep n =100(paper) scan in the map
#define mapsize 100 //100 10 2
//从每个scan list中采样的点数s=100
#define numselect 200 // 因为kitti一帧数据太多了 665 650 800（0.7%）222 807 657 230 200
#define SampNeiThr 0.20 //计算特征点和地图中最近点法线 的近邻范围 原来是0.5  考虑自适应的选择？
#define SampNeiNum 20 //尝试估计法线时用近邻个数选邻域
//采样中的 outlier 判别阈值
#define OutlierThr 0.20  //0.2；再调的参数！ 
#define RadThr 0.20 //计算I_x时那个box邻域半径
// #define numkneibor 5 //计算map 点法线 时近邻个数 5，8，9
//定义IMLS surface的参数 h
#define h_imls 0.06
//lossfunction 的阈值参数
#define lossarg 0.10 //huberloss 原始是0.1 0.2 
//ICP优化次数
#define numICP 21 //论文是20次 为了看20次ICP后可视化的结果 21
#define maxnumiter1ICP 4 //一次ICP中最大迭代次数（for ceres）
//保存不同k近邻下表示选择邻域大小的熵值
std::vector<float> Ef_k(9); //分为9分
//近邻大小的取值范围
std::vector<int> k_candi {4, 6, 8, 12, 20, 32, 47, 55, 64};
//表示float的无穷大
float infinity = 1.0 / 0.0;
//统一法线的方向 视点坐标的设置  会影响法线可视化
Eigen::Vector3d Viewpoint(0, 0, 0);

//距离滤波 最近 最远 阈值
#define distance_near_thresh 5 // loam 已经剔除了5m之内的点
#define distance_far_thresh 80 //待设置 kitti<81
//设置下采样的相关参数
#define downsample_method "VOXELGRID"  //APPROX_VOXELGRID
#define downsample_resolution 0.2 //0.1 0.2
//设置外点去除的相关参数
#define outlier_removal_method "STATISTICAL" //STATISTICAL RADIUS
#define radius_radius 0.5
#define radius_min_neighbors 5
#define statistical_mean_k 30
#define statistical_stddev 1.2

//small group removal 界限
#define Xbox 14
#define Ybox 14
#define Zbox 4
#define minclusterRMThs 20 //box之内的点多于该阈值 才被去除
#define maxclusterRMThs 150

//当前帧采样之后用于maping的点  9*100=900
pcl::PointCloud<pointxyzinormal>::Ptr CloudSampled(new pcl::PointCloud<pointxyzinormal>());

//input & output: points in one frame. local --> global 一帧的所有点
pcl::PointCloud<PointType>::Ptr laserCloudFullRes(new pcl::PointCloud<PointType>());
pcl::PointCloud<PointType>::Ptr Cloudprocess_raw(new pcl::PointCloud<PointType>());
pcl::PointCloud<PointType>::Ptr CloudProcessed(new pcl::PointCloud<PointType>());

//存放n=100个已经注册的之前的点云作为map
std::vector<pcl::PointCloud<pointxyzinormal>> ModelPointCloud(mapsize);
//从当前地图指针数组中提取的点
pcl::PointCloud<pointxyzinormal>::Ptr laserCloudFromMap(new pcl::PointCloud<pointxyzinormal>());

//kd-tree
pcl::KdTreeFLANN<pointxyzinormal>::Ptr kdtreeFromMap(new pcl::KdTreeFLANN<pointxyzinormal>());
pcl::KdTreeFLANN<pointxyzinormal>::Ptr kdtreeProcessed(new pcl::KdTreeFLANN<pointxyzinormal>());//用于计算当前scan中每个点的法线

//用于优化的变换矩阵
double parameters[7] = {0, 0, 0, 1, 0, 0, 0};
Eigen::Map<Eigen::Quaterniond> q_w_curr(parameters); //？ 前四个参数   
Eigen::Map<Eigen::Vector3d> t_w_curr(parameters + 4);
// wmap_T_odom * odom_T_curr = wmap_T_curr;
// transformation between odom's world and map's world frame
Eigen::Quaterniond q_wmap_wodom(1, 0, 0, 0); // w,x,y,z 地图的世界坐标和里程计的世界坐标两者是一致的
Eigen::Vector3d t_wmap_wodom(0, 0, 0);

//当前帧k相对于 odom world的变换
Eigen::Quaterniond q_wodom_curr(1, 0, 0, 0);
Eigen::Vector3d t_wodom_curr(0, 0, 0);
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
// std::queue<sensor_msgs::PointCloud2ConstPtr> GroundLastBuf; //缓存地面点的
std::queue<sensor_msgs::PointCloud2ConstPtr> fullResBuf;
std::queue<nav_msgs::Odometry::ConstPtr> odometryBuf;
std::mutex mBuf;

std::vector<int> pointSearchInd;
std::vector<float> pointSearchSqDis;
// std::vector<int> pointNeighborInd; 
// std::vector<float> pointNeighborSqDis;

PointType pointOri, pointSel;//原始点，变换后的点 XYZI
// pointxyz pointsave;//保存为pcd文件时的点的类型
pointxyzinormal pointWN, pointWNSel;//带有法线信息的点云类型 节省了地图中点法线的再次计算

ros::Publisher pubLaserCloudSurround, pubLaserCloudMap, pubLaserCloudFullRes, pubOdomAftMapped, pubLaserAfterMappedPath, pubCloudSampled, pubCloudbfPreProcess, pubCloudafPreProcess;
// ros::Publisher pubOdomAftMappedHighFrec;
//发布每类采样点 用于分析特征点的采样效果
std::vector<ros::Publisher> pubEachFeatList;
bool PUB_EACH_List = true;//用于调试 false
nav_msgs::Path laserAfterMappedPath;

ofstream outcloudscan;


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

Eigen::Vector3d X_axis(1, 0, 0);
Eigen::Vector3d Y_axis(0, 1, 0);
Eigen::Vector3d Z_axis(0, 0, 1);

//把XYZI类型转为XYZ
void XYZIToXYZ(PointType const *const pi, pointxyz *const po)
{
	po->x = pi->x;
	po->y = pi->y;
	po->z = pi->z; //! 致命的错误。。。 导致保存的点云被压扁！
}

//把XYZI类型转为XYZINormal
void XYZIToXYZINormal(PointType const *const pi, pointxyzinormal *const po)
{
	po->x = pi->x;
	po->y = pi->y;
	po->z = pi->z; //! 致命的错误。。。 导致保存的点云被压扁！
	po->intensity = pi->intensity;
	po->normal[0] = 0;
	po->normal[1] = 0;
	po->normal[2] = 0;
}

//把XYZINormal转为XYZI类型  发布前再转换回来
void XYZINormalToXYZI(pointxyzinormal const *const pi, PointType *const po)
{
	po->x = pi->x;
	po->y = pi->y;
	po->z = pi->z; //! 致命的错误。。。 导致保存的点云被压扁！
	po->intensity = pi->intensity;
}

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
//点的类型有变化！
void pointWNormalAssociateToMap(pointxyzinormal const *const pi, pointxyzinormal *const po)
{
	Eigen::Vector3d point_curr(pi->x, pi->y, pi->z);
	Eigen::Vector3d point_w = q_w_curr * point_curr + t_w_curr;
	po->x = point_w.x();
	po->y = point_w.y();
	po->z = point_w.z();
	po->intensity = pi->intensity;
	//po->intensity = 1.0;
	po->normal[0] = pi->normal[0];//法线信息保留不变
	po->normal[1] = pi->normal[1];
	po->normal[2] = pi->normal[2];
}
/*
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
*/
void laserCloudFullResHandler(const sensor_msgs::PointCloud2ConstPtr &laserCloudFullRes2)
{
	mBuf.lock();
	fullResBuf.push(laserCloudFullRes2);
	mBuf.unlock();
}
/*
//处理传来的地面点
void CloudGroundLastHandler(const sensor_msgs::PointCloud2ConstPtr &CloudGroundLast2)
{
	mBuf.lock();
	GroundLastBuf.push(CloudGroundLast2);
	mBuf.unlock();
}
*/

void CloudprocessedHandler(const sensor_msgs::PointCloud2ConstPtr &CloudProcessed2)
{
	mBuf.lock();
	processedBuf.push(CloudProcessed2);
	mBuf.unlock();
}
/*
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

	
	//装换为map world frame为参考
	// Eigen::Quaterniond q_w_curr = q_wmap_wodom * q_wodom_curr;
	// Eigen::Vector3d t_w_curr = q_wmap_wodom * t_wodom_curr + t_wmap_wodom; 

	
	// nav_msgs::Odometry odomAftMapped;
	// odomAftMapped.header.frame_id = "/camera_init";
	// odomAftMapped.child_frame_id = "/aft_mapped";
	// odomAftMapped.header.stamp = laserOdometry->header.stamp;
	// odomAftMapped.pose.pose.orientation.x = q_w_curr.x();
	// odomAftMapped.pose.pose.orientation.y = q_w_curr.y();
	// odomAftMapped.pose.pose.orientation.z = q_w_curr.z();
	// odomAftMapped.pose.pose.orientation.w = q_w_curr.w();
	// odomAftMapped.pose.pose.position.x = t_w_curr.x();
	// odomAftMapped.pose.pose.position.y = t_w_curr.y();
	// odomAftMapped.pose.pose.position.z = t_w_curr.z();
	// pubOdomAftMappedHighFrec.publish(odomAftMapped);
	
}
*/
//滤波1  留下距离lidar在 [distance_near_thresh, distance_far_thresh]的点 实测就少了不到15点 影响不大
pcl::PointCloud<PointType>::Ptr distance_filter(const pcl::PointCloud<PointType>::Ptr& cloud) 
{
    pcl::PointCloud<PointType>::Ptr filtered(new pcl::PointCloud<PointType>());
    filtered->reserve(cloud->size());
    // std::vector<double> dis_stat;
    std::copy_if(cloud->begin(), cloud->end(), std::back_inserter(filtered->points),
        [&](const PointType& p) {
        double d = p.getVector3fMap().norm();//getVector3fMap() 返回Eigen::Vector3d norm 得到2范数 就是距离 
        // dis_stat.push_back(d);
        return d > distance_near_thresh && d < distance_far_thresh;//点到lidar的距离 这个限制了整个点云的规模 去除过近或过远的点
        // return true; //经过实验统计nearst distance : 5.46376, far distance : 80.5128
        }
    );
    //得到最远/最近距离值
    // std::vector<double>::iterator mindis = min_element(dis_stat.begin(), dis_stat.end());
    // int min_ind = std::distance(dis_stat.begin(), mindis);
    // double dist_near = dis_stat[min_ind];
    
    // std::vector<double>::iterator maxdis = max_element(dis_stat.begin(), dis_stat.end());
    // int max_ind = std::distance(dis_stat.begin(), maxdis);
    // double dist_far = dis_stat[max_ind];
    // std::cout<<" nearst distance : "<<dist_near<<", far distance : "<<dist_far<<endl;

    filtered->width = filtered->size();
    filtered->height = 1;
    filtered->is_dense = false;

    filtered->header = cloud->header;

    return filtered;
}

//滤波2 对点云下采样
pcl::PointCloud<PointType>::Ptr downsample(const pcl::PointCloud<PointType>::Ptr& cloud) 
{
    pcl::Filter<PointType>::Ptr downsample_filter;//可指代前面提到的不同体素滤波方法
    // if(downsample_method == "VOXELGRID") 
	if( strcmp(downsample_method, "VOXELGRID") == 0)
    {//用质心
      //初始化体素下采样的相关函数
      std::cout << "downsample: VOXELGRID " << downsample_resolution << std::endl;
      boost::shared_ptr<pcl::VoxelGrid<PointType>> voxelgrid(new pcl::VoxelGrid<PointType>());
    //   pcl::VoxelGrid<PointType>::Ptr voxelgrid(new pcl::VoxelGrid<PointType>());
      voxelgrid->setLeafSize(downsample_resolution, downsample_resolution, downsample_resolution);
      downsample_filter = voxelgrid;
    } 
    // else if(downsample_method == "APPROX_VOXELGRID") 
	else if(strcmp(downsample_method, "APPROX_VOXELGRID") == 0)
    {//小立方体的中心来近似该立方体内的若干点。相比于VoxelGrid，计算速度稍快，但也损失了原始点云局部形态的精细度
      std::cout << "downsample: APPROX_VOXELGRID " << downsample_resolution << std::endl;
      boost::shared_ptr<pcl::ApproximateVoxelGrid<PointType>> approx_voxelgrid(new pcl::ApproximateVoxelGrid<PointType>());
    //   pcl::ApproximateVoxelGrid<PointType>::Ptr approx_voxelgrid(new pcl::ApproximateVoxelGrid<PointType>());
      approx_voxelgrid->setLeafSize(downsample_resolution, downsample_resolution, downsample_resolution);
      downsample_filter = approx_voxelgrid;
    } 
    else 
    {
    //   if(downsample_method != "NONE") {
	  if(strcmp(downsample_method, "NONE") != 0) {
        std::cerr << "warning: unknown downsampling type (" << downsample_method << ")" << std::endl;
        std::cerr << "       : use passthrough filter" <<std::endl;
      }
      std::cout << "downsample: NONE" << std::endl;
    }

    
    if(!downsample_filter) {
      return cloud;
    }

    pcl::PointCloud<PointType>::Ptr filtered(new pcl::PointCloud<PointType>());
    downsample_filter->setInputCloud(cloud);
    downsample_filter->filter(*filtered);
    filtered->header = cloud->header;

    return filtered;
}

//滤波3 去除点云outlier
pcl::PointCloud<PointType>::Ptr outlier_removal(const pcl::PointCloud<PointType>::Ptr& cloud)
{
    pcl::Filter<PointType>::Ptr outlier_removal_filter;
    // if(outlier_removal_method == "STATISTICAL") 
	if(strcmp(outlier_removal_method, "STATISTICAL") == 0)
    {//统计法
      int mean_k = statistical_mean_k;
      double stddev_mul_thresh = statistical_stddev;
      std::cout << "outlier_removal: STATISTICAL " << mean_k << " - " << stddev_mul_thresh << std::endl;

      pcl::StatisticalOutlierRemoval<PointType>::Ptr sor(new pcl::StatisticalOutlierRemoval<PointType>());
      sor->setMeanK(mean_k);// 创建滤波器，对每个点分析的临近点的个数设置为50 ，并将标准差的倍数设置为1  这意味着如果一
      //个点的距离超出了平均距离一个标准差以上，则该点被标记为离群点，并将它移除，存储起来
      sor->setStddevMulThresh(stddev_mul_thresh);
      outlier_removal_filter = sor;
    } 
    // else if(outlier_removal_method == "RADIUS") 
	else if(strcmp(outlier_removal_method, "RADIUS") == 0)
    {//半径
      double radius = radius_radius;
      int min_neighbors = radius_min_neighbors;
      std::cout << "outlier_removal: RADIUS " << radius << " - " << min_neighbors << std::endl;

      pcl::RadiusOutlierRemoval<PointType>::Ptr rad(new pcl::RadiusOutlierRemoval<PointType>());
      rad->setRadiusSearch(radius);//在每点r半径内搜索邻域点个数,若小于min_neighbors就去除
      rad->setMinNeighborsInRadius(min_neighbors);
      outlier_removal_filter = rad;
    } 
    else 
    {
      std::cout << "outlier_removal: NONE" << std::endl;
    }

    if(!outlier_removal_filter) {
      return cloud;
    }

    pcl::PointCloud<PointType>::Ptr filtered(new pcl::PointCloud<PointType>());
    outlier_removal_filter->setInputCloud(cloud);
    outlier_removal_filter->filter(*filtered);
    filtered->header = cloud->header;

    return filtered;
}

//将src分离为地面点和非地面点
void seperate_ground( const pcl::PointCloud<PointType>::Ptr& src_cloud, pcl::PointCloud<PointType>::Ptr& ground_cloud, pcl::PointCloud<PointType>::Ptr& out_cloud )
{
    pcl::ModelCoefficients::Ptr coefficients (new pcl::ModelCoefficients);
    pcl::PointIndices::Ptr inliers (new pcl::PointIndices);
    // Create the segmentation object
    pcl::SACSegmentation<PointType> segground;
    segground.setOptimizeCoefficients (true);//可选
    segground.setModelType (pcl::SACMODEL_PLANE);//设置分割模型类别:平面
    segground.setMethodType (pcl::SAC_RANSAC);//设置用哪个随机参数估计方法
    segground.setDistanceThreshold (0.15);//点到估计模型的距离最大值 可调节 0.01 0.05 0.1 0.15 0.17 0.20 0.19
    segground.setMaxIterations (100);  //设置最大迭代次数 否则默认50

    segground.setInputCloud (src_cloud); //待分割的点云
    segground.segment (*inliers, *coefficients);
    if (inliers->indices.size () == 0)
    {
        PCL_ERROR ("Could not estimate a planar model for the given cloud frame.");
    }

    // 提取地面
    pcl::ExtractIndices<PointType> extract;//创建点云提取对象
    extract.setInputCloud (src_cloud);//设置输入点云
    extract.setIndices (inliers);//设置分割后的内点（地面点）为需要提取的点集
    extract.setNegative (false); //设置提取内点而非外点
    extract.filter (*ground_cloud);//提取输出存储到laserCloudGround

    // 提取除地面外的点云
    extract.setNegative (true);//提取inlier之外的点
    extract.filter (*out_cloud);//提取输出存储到laserCloudWOGround
    
    std::cout <<"the size of Ground point is " << ground_cloud->points.size () << ", and the size of other is " << out_cloud->points.size() << endl;


}

//通过聚类 去除动态物体
pcl::PointCloud<PointType>::Ptr dynamic_object_removal(const pcl::PointCloud<PointType>::Ptr& src_cloud)
{
    //聚类后的所有点云团 待输出
    pcl::PointCloud<PointType>::Ptr cloud_cluster_all(new pcl::PointCloud<PointType>());
    pcl::search::KdTree<PointType>::Ptr stree (new pcl::search::KdTree<PointType>);
    stree->setInputCloud (src_cloud);//　不含地面的点云
    std::vector<pcl::PointIndices> cluster_indices;// 点云团索引
    pcl::EuclideanClusterExtraction<PointType> ec;// 欧式聚类对象
    ec.setClusterTolerance (0.90); // 设置近邻搜索的搜索半径为0.5m paper的参数
    ec.setMinClusterSize (6);  // 设置一个聚类需要的最少的点数目为100 不设限(default=1)，因为要保留出动态物体点云团之外的所有点
    ec.setMaxClusterSize (25000); // 设置一个聚类需要的最大点数目为1000
    ec.setSearchMethod (stree); // 设置点云的搜索机制
    ec.setInputCloud (src_cloud);
    ec.extract (cluster_indices);           //从点云中提取聚类，并将点云索引保存在cluster_indices中
    
    //迭代访问点云索引cluster_indices,直到分割处所有聚类
    int j = 0;
    for (std::vector<pcl::PointIndices>::const_iterator it = cluster_indices.begin (); it != cluster_indices.end (); ++it)
    {
        pcl::PointCloud<PointType>::Ptr cloud_cluster (new pcl::PointCloud<PointType>());//每个group
        for (std::vector<int>::const_iterator pit = it->indices.begin (); pit != it->indices.end (); ++pit)
        {   
            cloud_cluster->points.push_back (src_cloud->points[*pit]); //获取每一个点云团　的　点
        }
        cloud_cluster->width = cloud_cluster->points.size ();
        cloud_cluster->height = 1;
        cloud_cluster->is_dense = true;
        // std::cout << "the " << j << "th cloud cluster point number:" << cloud_cluster->points.size() << '\n';

        //计算该group的bounding box x,y,z
        PointType minPt, maxPt;
        pcl::getMinMax3D (*cloud_cluster, minPt, maxPt);
        float cloud_cluster_x = maxPt.x - minPt.x;
        float cloud_cluster_y = maxPt.y - minPt.y;
        float cloud_cluster_z = maxPt.z - minPt.z;

        // std::cout << "x range: "<< cloud_cluster_x << ", y range: " << cloud_cluster_y << ", z range: " << cloud_cluster_z << '\n';

        //若该group小于指定的box，不保留，跳过,判断下一个 points group 
        //关于如何准确的聚类到移动物体，判断条件可能需要调整  size界限越大，保留地越多
        if ( cloud_cluster_x <= Xbox && cloud_cluster_y <= Ybox && cloud_cluster_z <= Zbox && 
            (cloud_cluster->points.size() >= minclusterRMThs) && (cloud_cluster->points.size() <= maxclusterRMThs) )
        // if ( cloud_cluster_x <= Xbox && cloud_cluster_y <= Ybox && cloud_cluster_z <= Zbox )
        {
            // std::cout << "the " << j << "th cloud cluster are removed !" << '\n';
            j++;
            continue;
        }
        j++;
        //保存到总点云
        *cloud_cluster_all += *cloud_cluster;
        
    }

    return cloud_cluster_all;
}

//计算法线并保存为xyzinormal格式
pcl::PointCloud<pointxyzinormal>::Ptr compute_normal( const pcl::PointCloud<PointType>::Ptr& src_cloud )
{
    pcl::PointCloud<pointxyzinormal>::Ptr dst_cloud(new pcl::PointCloud<pointxyzinormal>);
    //当前点云建立kdtree
    pcl::KdTreeFLANN<PointType>::Ptr kdtreeFromsrc(new pcl::KdTreeFLANN<PointType>());
    kdtreeFromsrc->setInputCloud(src_cloud);
    std::vector<int> pointShInd;
    std::vector<float> pointShSqDis;
    int nump = src_cloud->points.size();

    for (int i=0; i<nump; i++)
    {
        PointType pointtmp = src_cloud->points[i];
        
        //对于当前点在scan中搜索指定半径内的近邻
		int numkdtree0 = kdtreeFromsrc->radiusSearch(pointtmp, SampNeiThr, pointShInd, pointShSqDis);
        if(numkdtree0 < 5)//若0.2m半径内找到的近邻太少 就使用近邻个数 or选择最优个数
        {
            pointShInd.clear();
            pointShSqDis.clear();
            // std::cout << " within 0.2m find " << numkdtree0 << " (<5)at points " << i <<endl;
            int numkdtree = kdtreeFromsrc->nearestKSearch(pointtmp, SampNeiNum, pointShInd, pointShSqDis); //按近邻个数搜索
            
            if (numkdtree < 3)//至少3点确定一平面
            {
                std::cout<<"less than 3 points find at point "<< i <<endl;
            }
        }
        /*
        //先遍历10个邻域半径 选择最优半径
        for (int ks = 0; ks < 9; ks++)
        {
            int kv = k_candi[ks];
            //对于当前点在scan中搜索指定半径Rj内的近邻
            int numkdtree0 = kdtreeFromsrc->nearestKSearch(pointtmp, kv, pointShInd, pointShSqDis);
            if (numkdtree0 < 3)//至少3点才能确定一平面
            {
                pointShInd.clear();
                pointShSqDis.clear();
                Ef_k[ks] = infinity;//给个无穷大
                continue;//找不够三点,该半径不可取
            }
            int numneighbor = pointShInd.size();//得到的半径内近邻个数
            std::vector<Eigen::Vector3d> neighbors;//存储若干近邻点
            Eigen::Vector3d center(0, 0, 0);//初始化近邻点的重心
            for (int j = 0; j < numneighbor; j++)
            {
                Eigen::Vector3d tmp(src_cloud->points[pointShInd[j]].x,
                                    src_cloud->points[pointShInd[j]].y,
                                    src_cloud->points[pointShInd[j]].z);
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

            double a1d = (lamda_1 - lamda_2) / lamda_1;
            double a2d = (lamda_2 - lamda_3) / lamda_1;
            double a3d = lamda_3 / lamda_1;

            Ef_k[ks] = -a1d * log(a1d) - a2d * log(a2d) - a3d * log(a3d);

            pointShInd.clear();
            pointShSqDis.clear();

        }
        //选择最小Ef值对应的近邻个数作为最优近邻大小
        std::vector<float>::iterator minEf = min_element(Ef_k.begin(), Ef_k.end());
        int minE_ind = std::distance(Ef_k.begin(), minEf);
        int K_opt = k_candi[minE_ind];//变为自己设置的可选值
        int numkdtree = kdtreeFromsrc->nearestKSearch(pointtmp, K_opt, pointShInd, pointShSqDis);
        */
        
		int numneighbor = pointShInd.size();//得到的半径内近邻个数
		std::vector<Eigen::Vector3d> neighbors;//存储若干近邻点
		Eigen::Vector3d center(0, 0, 0);//初始化近邻点的重心
        for (int j = 0; j < numneighbor; j++)
        {
            Eigen::Vector3d tmp(src_cloud->points[pointShInd[j]].x,
                                src_cloud->points[pointShInd[j]].y,
                                src_cloud->points[pointShInd[j]].z);
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

        //获取最小特征值对应的特征向量 即为法线
        Eigen::Vector3d rawnormcurr = saes.eigenvectors().col(0);
        //还从没出现过该情况
        if(std::isnan(rawnormcurr.x()) || std::isnan(rawnormcurr.y()) || std::isnan(rawnormcurr.z()))//
        {
            // std::cout <<"src_cloud NORM NaN Warning!"<<endl;
            printf("src_cloud norm NaN occur at %d th sampled points ",i );	
        }

        Eigen::Vector3d pointcurr(pointtmp.x, pointtmp.y, pointtmp.z);

        Eigen::Vector3d pointnormcurr = rawnormcurr.normalized();//归一化
        //更改视点  使得更一致 可视化更直观 
        if( pointnormcurr.dot(pointcurr - Viewpoint) > 0)
        {//设置法线方向统一
            pointnormcurr = -pointnormcurr;
        }
        
        pointxyzinormal pointout;
        pointout.x = pointtmp.x;
        pointout.y = pointtmp.y;
        pointout.z = pointtmp.z;
        // pointout.intensity = 0;
		pointout.intensity = pointtmp.intensity;//它本身inyensity是有值的
        pointout.normal[0] = pointnormcurr.x();
        pointout.normal[1] = pointnormcurr.y();
        pointout.normal[2] = pointnormcurr.z();
        dst_cloud->push_back(pointout);

        pointShInd.clear();
        pointShSqDis.clear();

    }
    return dst_cloud; 

}

//计算a2d并计算特征值 法线已经前面计算过了
void compute_feature( const pcl::PointCloud<pointxyzinormal>::Ptr& src_cloud )
{
    //当前点云建立kdtree
    pcl::KdTreeFLANN<pointxyzinormal>::Ptr kdtreeFromsrc(new pcl::KdTreeFLANN<pointxyzinormal>());
    kdtreeFromsrc->setInputCloud(src_cloud);
    std::vector<int> pointShInd;
    std::vector<float> pointShSqDis;
    int nump = src_cloud->points.size();
    for (int i=0; i<nump; i++)
    {
        pointxyzinormal pointtmp = src_cloud->points[i];
        Eigen::Vector3d pointcurr(pointtmp.x,
                                  pointtmp.y,
                                  pointtmp.z);
        //先遍历10个邻域半径 选择最优半径
        for (int ks = 0; ks < 9; ks++)
        {
            int kv = k_candi[ks];
            //对于当前点在scan中搜索指定半径Rj内的近邻
            int numkdtree0 = kdtreeFromsrc->nearestKSearch(pointtmp, kv, pointShInd, pointShSqDis);
            if (numkdtree0 < 3)//至少3点才能确定一平面
            {
                pointShInd.clear();
                pointShSqDis.clear();
                Ef_k[ks] = infinity;//给个无穷大
                continue;//找不够三点,该半径不可取
            }
            int numneighbor = pointShInd.size();//得到的半径内近邻个数
            std::vector<Eigen::Vector3d> neighbors;//存储若干近邻点
            Eigen::Vector3d center(0, 0, 0);//初始化近邻点的重心
            for (int j = 0; j < numneighbor; j++)
            {
                Eigen::Vector3d tmp(src_cloud->points[pointShInd[j]].x,
                                    src_cloud->points[pointShInd[j]].y,
                                    src_cloud->points[pointShInd[j]].z);
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

            double a1d = (lamda_1 - lamda_2) / lamda_1;
            double a2d = (lamda_2 - lamda_3) / lamda_1;
            double a3d = lamda_3 / lamda_1;

            Ef_k[ks] = -a1d * log(a1d) - a2d * log(a2d) - a3d * log(a3d);

            pointShInd.clear();
            pointShSqDis.clear();

        }
        //选择最小Ef值对应的近邻个数作为最优近邻大小
        std::vector<float>::iterator minEf = min_element(Ef_k.begin(), Ef_k.end());
        int minE_ind = std::distance(Ef_k.begin(), minEf);
        int K_opt = k_candi[minE_ind];//变为自己设置的可选值
        int numkdtree = kdtreeFromsrc->nearestKSearch(pointtmp, K_opt, pointShInd, pointShSqDis);
		if (numkdtree < 3)//至少3点确定一平面
		{
			std::cout<<"less than 3 points find at point "<< i <<endl;
		}
        int numneighbor = pointShInd.size();//得到的半径内近邻个数
		std::vector<Eigen::Vector3d> neighbors;//存储若干近邻点
		Eigen::Vector3d center(0, 0, 0);//初始化近邻点的重心
        for (int j = 0; j < numneighbor; j++)
        {
            Eigen::Vector3d tmp(src_cloud->points[pointShInd[j]].x,
                                src_cloud->points[pointShInd[j]].y,
                                src_cloud->points[pointShInd[j]].z);
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

        double a2d = (lamda_2 - lamda_3)/lamda_1;

        //得到保存的法线
        Eigen::Vector3d normcurr( pointtmp.normal[0],
                                  pointtmp.normal[1],
                                  pointtmp.normal[2]);
        if(std::isnan(normcurr.x()) || std::isnan(normcurr.y()) || std::isnan(normcurr.z()))//
        {
            printf("src_cloud norm NaN occur at %d th points ",i );	
        }
        
        //分别计算当前点的9个特征 并保存在对应数组中 使用了前面计算好的a2d数组
        Eigen::Vector3d tmpcross = pointcurr.cross(normcurr);
        samplefeature1[i] = (tmpcross.dot(X_axis)) * a2d * a2d;
        samplefeature2[i] = -samplefeature1[i];
        samplefeature3[i] = (tmpcross.dot(Y_axis)) * a2d * a2d;
        samplefeature4[i] = -samplefeature3[i];
        samplefeature5[i] = (tmpcross.dot(Z_axis)) * a2d * a2d;
        samplefeature6[i] = -samplefeature5[i];
        samplefeature7[i] = fabs(normcurr.dot(X_axis)) * a2d * a2d;
        samplefeature8[i] = fabs(normcurr.dot(Y_axis)) * a2d * a2d;
        samplefeature9[i] = fabs(normcurr.dot(Z_axis)) * a2d * a2d;
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

        //清空之前pointSearchInd, pointShSqDis
        pointShInd.clear();
        pointShSqDis.clear();
    }

}

//IMLS中的采样以及mapping过程（借鉴下面的void process() ）
void imls()
{
	while(1)
	{

		// while ( !fullResBuf.empty()  && !processedBuf.empty())
		while ( !fullResBuf.empty() )
		{
			mBuf.lock();
			//确保各容器数据的时间戳是合理的
			//待采样的cloud
			// while (!processedBuf.empty() && processedBuf.front()->header.stamp.toSec() < fullResBuf.front()->header.stamp.toSec())
			// 	processedBuf.pop();
			// if (processedBuf.empty())
			// {
			// 	mBuf.unlock();
			// 	break;//无数据，重新接收，并执行判断是否接收到
			// }

			//得到时间戳
			// timecloudprocessed = processedBuf.front()->header.stamp.toSec();
			timeLaserCloudFullRes = fullResBuf.front()->header.stamp.toSec();

			//确保各类点云的时间戳同步（那前面为啥也判断了时间？）
			// if (timecloudprocessed != timeLaserCloudFullRes /*||
			// 	timecloudGround != timeLaserCloudFullRes ||
			// 	timeLaserOdometry != timeLaserCloudFullRes*/)
			// {
			// 	printf("time full %f processed %f \n", timeLaserCloudFullRes, timecloudprocessed);
			// 	printf("unsync messeage!");
			// 	mBuf.unlock();
			// 	break;
			// }

			CloudProcessed->clear();
			if(int(CloudProcessed->points.size()) != 0)
			{
				std::cout<<"Failed to clear CloudProcessed !\n"<<endl;
			}
			CloudSampled.reset(new pcl::PointCloud<pointxyzinormal>());
			if(int(CloudSampled->points.size()) != 0)
			{
				std::cout<<"Failed to clear CloudSampled !\n"<<endl;
			}
			laserCloudFullRes->clear(); //在debug阶段 它和cloudprocessed其实一样
			if(int(laserCloudFullRes->points.size()) != 0)
			{
				std::cout<<"Failed to clear laserCloudFullRes !\n"<<endl;
			}
			pcl::fromROSMsg(*fullResBuf.front(), *laserCloudFullRes);
			fullResBuf.pop();

			// Cloudprocess_raw->clear();
			// pcl::fromROSMsg(*processedBuf.front(), *Cloudprocess_raw);
			// processedBuf.pop();


			// 位姿的初始估计值 需要知道前两帧的优化后的最终位姿
			if(frameCount < 2 )
			{//第0帧，第1帧 位姿初始值都为I
				//第0帧 直接输出单位阵 q_wodom_curr初始值就是I
				if (frameCount == 1)
				{//debug 先用一个粗略的初始化测试
					Eigen::Matrix3d R_f1;
					// R_f1 << 0.999999, -0.000207, 0.001332, 0.000208, 0.9999996, -0.000895, -0.001332, 0.000895, 0.9999987;
					R_f1 << 0.99, -0.00, 0.00, 0.00, 0.99, -0.00, -0.00, 0.00, 0.99;
					Eigen::Quaterniond q_f1(R_f1);
					// Eigen::Vector3d t_f1(1.3107573, 0.0077851, 0.0157089); 1.310828, -0.003942, 0.010135
					Eigen::Vector3d t_f1(0.50, -0.001, 0.005); //1.10, 0.00001, 0.00001
					q_wodom_curr = q_f1;
					t_wodom_curr = t_f1;
				}
				std::cout<<"the init 'wodom_curr' pose of frame"<<frameCount<<": q= "<<q_wodom_curr.coeffs().transpose()<<" t= "<<t_wodom_curr.transpose()<<endl;
			
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
				// t_wodom_k_1 = q_wmap_wodom.inverse() * (t_w_k_1 - t_wmap_wodom);
				q_wodom_k_1 = q_w_k_1;
				t_wodom_k_1 = t_w_k_1;

				// q_wodom_k_2 = q_wmap_wodom.inverse() * q_w_k_2;
				// t_wodom_k_2 = q_wmap_wodom.inverse() * (t_w_k_2 - t_wmap_wodom);
				q_wodom_k_2 = q_w_k_2;
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

				std::cout<<"the init 'wodom_curr' pose of frame"<<frameCount<<": q= "<<q_wodom_curr.coeffs().transpose()<<" t= "<<t_wodom_curr.transpose()<<endl;

				
			}
			
			mBuf.unlock();

			TicToc t_whole;

			//把线性变换后的位姿转为相对于map世界坐标系的位姿 （初始时实际上乘的是单位矩阵）
			// transformAssociateToMap();//第一帧的话 应该有q_w_curr=1 0 0 0
			q_w_curr = q_wodom_curr;
			t_w_curr = t_wodom_curr;
			std::cout<<"the init 'w_curr' pose of frame"<<frameCount<<": q= "<<q_w_curr.coeffs().transpose()<<" t= "<<t_w_curr.transpose()<<"\n"<<endl;

			//----------------预处理---------------------//
			TicToc t_prefilter;
			int numprocess_raw = laserCloudFullRes->points.size();
			//1 距离滤波
			pcl::PointCloud<PointType>::Ptr dist_filter(new pcl::PointCloud<PointType>());
			std::cout<<"distance filtering Scan "<<frameCount<<" ..."<<endl;
			dist_filter = distance_filter(laserCloudFullRes);
			int numdistf = dist_filter->points.size();
    		std::cout<<"before prefilter: "<<numprocess_raw<<" ,after dist_filter: "<<numdistf<<endl;

			//2.下采样
			pcl::PointCloud<PointType>::Ptr DS_filter(new pcl::PointCloud<PointType>());
			std::cout<<"Downsampling Scan "<<frameCount<<" ..."<<endl;
			DS_filter = downsample(dist_filter);
			int numds = DS_filter->points.size();
    		std::cout<<"after downsampled: "<<numds<<endl;

			//3.去除离群点
			pcl::PointCloud<PointType>::Ptr Outlierm(new pcl::PointCloud<PointType>());
			std::cout<<"Removing Outlier from Scan "<<frameCount<<" ..."<<endl;
			Outlierm = outlier_removal(DS_filter); //从这里拿特征点？
			int numoutlierrm = Outlierm->points.size();
			std::cout<<"after Outliers removed: "<<numoutlierrm<<endl;

			printf("Scan point cloud prefiltering time %f ms\n", t_prefilter.toc());
			//-----------------动态物体去除-------------------------//
			TicToc t_dynamicobjectremoval;
			//1.分离出地面点和其余非地面点
			//提取的地面点
			pcl::PointCloud<PointType>::Ptr Ground_cloud(new pcl::PointCloud<PointType>());
			//去除地面的点云
			pcl::PointCloud<PointType>::Ptr WOGround(new pcl::PointCloud<PointType>());
			std::cout<<"seperating Ground from Scan "<<frameCount<<" ..."<<endl;
			seperate_ground(Outlierm, Ground_cloud, WOGround);
			int numWOG = WOGround->points.size();
			std::cout<<"after Seperating Ground points: "<<numWOG<<endl;

			//2.聚类 去除 指定边界box之内的点云
			pcl::PointCloud<PointType>::Ptr rmDynamic(new pcl::PointCloud<PointType>());
			std::cout<<"Remove Dynamic Objects from Scan_WOGround "<<frameCount<<" ..."<<endl;
			rmDynamic = dynamic_object_removal(WOGround);
			int numRMdynamic = rmDynamic->points.size();
			std::cout<<"after Removed Dynamic Objects: "<<numRMdynamic<<endl;

			printf("Dynamic Objects Removal time %f ms\n", t_dynamicobjectremoval.toc());
			//增加回地面点
			*rmDynamic += *Ground_cloud;
			*CloudProcessed = *rmDynamic;
			int numProcessed = CloudProcessed->points.size();//经历预滤波和运动物体取出后的点数
			std::cout<<"Before All-Preprocess Scan "<<frameCount<<": "<<numprocess_raw<<" ,after: "<<numProcessed<<endl;
			
			TicToc t_getfeatv;
			// pcl::PointXYZI PointType
			//先计算当前帧点云的法线 数据格式变为pointxyzinormal
			pcl::PointCloud<pointxyzinormal>::Ptr ScanWN(new pcl::PointCloud<pointxyzinormal>());
			
			std::cout<<"Computing normal for Scan-Processed "<<frameCount<<" ..."<<endl;
			//r0.2 k20
   	 		ScanWN = compute_normal(CloudProcessed); //使用的是一些列预处理后的scan current
			std::cout<<"Computing 9 feature for Scan-Processed "<<frameCount<<" ..."<<endl;
			//k opt [4,64] 从9个值中挑选
			compute_feature(ScanWN);//点数还是numProcessed!
			printf("point features compute time %f ms\n", t_getfeatv.toc());

			/*
			//保存当前帧为pcd文件 已经包含了计算的法线信息 之后打开直接可视化
			pcl::PointCloud<pointxyzinormal> cloudScansave;
			
			for (int i = 0; i < numProcessed; i++)
			{
				pointWN = ScanWN->points[i];
				cloudScansave.push_back(pointWN);
			}
			//写入文件
			pcl::io::savePCDFileASCII("/home/dlr/imlslam/pcdsave/scanWN_" + std::to_string(frameCount) + ".pcd", cloudScansave);
			std::cerr<<"Saved "<<cloudScansave.points.size()<<" points to scanWN_"<<std::to_string(frameCount)<<".pcd"<<endl;
			*/
			//对9个表进行从大到小排序
			std::sort (cloudSortInd1, cloudSortInd1 + numProcessed, comp1);
			std::sort (cloudSortInd2, cloudSortInd2 + numProcessed, comp2);
			std::sort (cloudSortInd3, cloudSortInd3 + numProcessed, comp3);
			std::sort (cloudSortInd4, cloudSortInd4 + numProcessed, comp4);
			std::sort (cloudSortInd5, cloudSortInd5 + numProcessed, comp5);
			std::sort (cloudSortInd6, cloudSortInd6 + numProcessed, comp6);
			std::sort (cloudSortInd7, cloudSortInd7 + numProcessed, comp7);
			std::sort (cloudSortInd8, cloudSortInd8 + numProcessed, comp8);
			std::sort (cloudSortInd9, cloudSortInd9 + numProcessed, comp9);

			laserCloudFromMap->clear();//更改了类型
			//从地图数组中得到当前所有modle point
			for (int i = 0; i < mapsize; i++)//100
			{
				*laserCloudFromMap += ModelPointCloud[i];//
			}
			//得到当前所有地图点 进行下采样： 
			int laserCloudMapNum = laserCloudFromMap->points.size();
			/*
			//没必要保存先
			if (frameCount > 0)//第0帧时候地图是空的
			{
				//保存当前地图点为pcd文件
				pcl::PointCloud<pointxyzinormal> cloudMapsave;
				
				for (int i = 0; i < laserCloudMapNum; i++)
				{
					pointWN = laserCloudFromMap->points[i];
					//先把所有点转为XYZ类型
					// XYZIToXYZ(&pointOri, &pointsave);
					cloudMapsave.push_back(pointWN); 
				}
				//写入文件
				pcl::io::savePCDFileASCII("/home/dlr/imlslam/pcdsave/mapmodelWN_" + std::to_string(frameCount) + ".pcd", cloudMapsave);
				std::cerr<<"Saved "<<cloudMapsave.points.size()<<" points to mapmodelWN_"<<std::to_string(frameCount)<<".pcd"<<endl;

			}
			*/

			/*
			//保存当前帧的采样特征点为pcd文件 已有法线信息
			pcl::PointCloud<pointxyzinormal> ScanSampledFeatsave;
			for (int i = 0; i < numscansampled; i++)
			{
				pointWN = CloudSampled->points[i];
				pointWNormalAssociateToMap(&pointWN, &pointWNSel);//存储的是已经做过变换到世界坐标的当前帧特征点
				// XYZIToXYZ(&pointSel, &pointsave);//先把所有点转为XYZ类型
				ScanSampledFeatsave.push_back(pointWNSel); 
			}
			//写入文件
			pcl::io::savePCDFileASCII("/home/dlr/imlslam/pcdsave/Feat_scanWN_" + std::to_string(frameCount) + ".pcd", ScanSampledFeatsave);
			std::cerr<<"Saved "<<ScanSampledFeatsave.points.size()<<" points to Feat_scanWN_"<<std::to_string(frameCount)<<".pcd"<<endl;
			*/

			/*
			//保存地图里的点，已经固定了的
			if (frameCount > 0)
			{
				// char file_name_model[256];
				// sprintf(file_name_model,"/home/dlr/imlslam/ICPmodel_" + frameCount + ".txt"); //file_name=ICPmodel_1.txt
				outcloudmodel.open("/home/dlr/imlslam/ICPmodel_" + std::to_string(frameCount) + ".txt");
				//先下采样 再写入文件
				pcl::PointCloud<pointxyzinormal>::Ptr tmpDS_map_0(new pcl::PointCloud<pointxyzinormal>());//下采样后的点云
				// pcl::VoxelGrid<PointType> downSizeFilter_map_0;
				// downSizeFilter_map_0.setInputCloud(laserCloudFromMap);
				//由于在得到cloudfrommap时已经进行了下采样，这里就几乎不再下采样了
				// downSizeFilter_map_0.setLeafSize(0.1, 0.1, 0.1);//可调参数 值越大，采样后越稀疏
				// downSizeFilter_map_0.filter(*tmpDS_map_0);
				*tmpDS_map_0 = *laserCloudFromMap;
				int numtmpDS_map_0 = tmpDS_map_0->points.size();
				std::cout<<"before: "<<int(laserCloudFromMap->points.size())<<" after model downsampled: "<<numtmpDS_map_0<<endl;
				for (int i = 0; i < numtmpDS_map_0; i++)
				{
					pointWN = tmpDS_map_0->points[i];
					Eigen::Vector3d pointm(pointWN.x, pointWN.y, pointWN.z);		
					// x y z r g b  pointx 认为是红色
					outcloudmodel << pointm.x() << " " <<pointm.y() << " " << pointm.z()<< endl;	 
				}
				outcloudmodel.close();
			}
			*/
			//地图中的特征点数目满足要求  若是第0帧 不执行这部分 包含匹配以及优化
			std::vector<pcl::PointCloud<pointxyzinormal>> CloudSampledFeat(9);//9类特征点云数组
			if (laserCloudMapNum > 10)
			{
				printf("points size of current map model: %d before mapping Frame %d \n", laserCloudMapNum, frameCount);//输出当前地图的大小
				TicToc t_tree;
				//建立model points kd tree
				kdtreeFromMap->setInputCloud(laserCloudFromMap);
				printf("build tree of map time %f ms \n", t_tree.toc());//建立地图点kdtree的时间
				
				TicToc t_opt;
				//ceres优化求解 迭代20次 mapping 过程
				//！to do 替换CERES，按照IMLS论文，根据<Linear Least-Squares Optimization for Point-to-Plane ICP Surface Registration>
				//改为近似的线性最小二乘求解
				
				for (int iterCount = 0; iterCount < numICP; iterCount++)  //debug
				{
					
					if (frameCount == 1 && (iterCount == 0 || iterCount == numICP-1))
					{
						// char file_name_feature[256];
						// sprintf(file_name_feature,"/home/dlr/imlslam/ICPfeat_%02d.txt",iterCount); //file_name=ICPfeat_01.txt
						// // outfile.open("/home/dlr/imlslam/" + file_name);
						// outfile.open(file_name_feature);

						char file_name_scan[256];
						sprintf(file_name_scan,"/home/dlr/imlslam/ICPscan1_%02d.txt",iterCount); //file_name=ICPscan1_01.txt
						outcloudscan.open(file_name_scan);
					}
					/*
					//每次迭代 都要做下面的事：
					//重新改回非线性优化优化模块
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
					*/
					
					TicToc t_scansample;
					for (int i = 0; i < 9; i++)
					{
						CloudSampledFeat[i].clear();//将所有列表采样点清零
						if(int(CloudSampledFeat[i].points.size()) != 0)
						{
							std::cout<<"Failed to clear CloudSampledFeat !\n"<<endl;
						}
					}
					//清空之前存储的先前帧的采样点
					CloudSampled.reset(new pcl::PointCloud<pointxyzinormal>());
					if(int(CloudSampled->points.size()) != 0)
					{
						std::cout<<"Failed to clear CloudSampled !\n"<<endl;
					}
					//把特征点采样移到这里 来判断是否外点  因为迭代中位姿更新 所以外点的判别也应是不断更新的
					int numpicked = 0;//计数
					for (int i = 0; i < numProcessed; i++)//遍历所有点 直到找满100个点
					{
						int ind1 = cloudSortInd1[i];//值较大点的索引
						//outlierrm还是xyzi格式
						// pointOri = Outlierm->points[ind1]; 
						// XYZIToXYZINormal(&pointOri, &pointWN);
						pointWN = ScanWN->points[ind1];
						// Eigen::Vector3d pointf(pointWN.x, pointWN.y, pointWN.z);
						pointWNormalAssociateToMap(&pointWN, &pointWNSel);//将该采样点转为map世界坐标系  使用的是odom的结果位姿 这步转换是否需要？
						//在现有地图点中找到距离采样点最近的一个点
						int numkdtreeo = kdtreeFromMap->nearestKSearch(pointWNSel, 1, pointSearchInd, pointSearchSqDis);
						if (numkdtreeo<=0)//没出现过 总能找到一个最近点
						{
							std::cerr<<"Error: No nearest point in map !"<<endl;
							
						}
						if (sqrt(pointSearchSqDis[0]) > OutlierThr)//此两点之间距离太大，认为是野点 
						{	//打印出最小的距离，用于调整阈值大小
							// std::cout<<"#1 outlier ! mindist="<<sqrt(pointSearchSqDis[0])<<endl;
							// outlierfeat_1 << pointf.x() << " " <<pointf.y() << " " << pointf.z()<< endl;
							pointSearchInd.clear();
                			pointSearchSqDis.clear();
							continue;//跳过，下个点
						}
						CloudSampledFeat[0].push_back(pointWN);
						// outfeat_1 << pointf.x() << " " <<pointf.y() << " " << pointf.z()<< endl;
						numpicked++;
						pointSearchInd.clear();
                		pointSearchSqDis.clear();
						if (numpicked >= numselect)//若已经选够100点 结束循环
						{
							break;
						}
						
					}

					numpicked = 0;//计数器清零
					for (int i = 0; i < numProcessed; i++)//遍历所有点 直到找满100个点
					{
						int ind2 = cloudSortInd2[i];//值较大点的索引
						// pointOri = Outlierm->points[ind2]; 
						// XYZIToXYZINormal(&pointOri, &pointWN);
						pointWN = ScanWN->points[ind2];
						pointWNormalAssociateToMap(&pointWN, &pointWNSel);//将该采样点转为map世界坐标系  使用的是odom的结果位姿
						//在现有地图点中找到距离采样点最近的一个点
						int numkdtreeo = kdtreeFromMap->nearestKSearch(pointWNSel, 1, pointSearchInd, pointSearchSqDis);
						if (numkdtreeo<=0)//没出现过 总能找到一个最近点
						{
							std::cerr<<"Error: No nearest point in map !"<<endl;
							
						}
						if (sqrt(pointSearchSqDis[0]) > OutlierThr)//此两点之间距离太大，认为是野点 
						{
							pointSearchInd.clear();
                			pointSearchSqDis.clear();
							continue;//跳过，下个点
						}
						
						CloudSampledFeat[1].push_back(pointWN);
						// outfeat_2 << pointf.x() << " " <<pointf.y() << " " << pointf.z()<< endl;
						numpicked++;
						pointSearchInd.clear();
                		pointSearchSqDis.clear();
						if (numpicked >= numselect)//若已经选够100点 结束循环
						{
							break;
						}	
					}
					numpicked = 0;//计数器清零
					for (int i = 0; i < numProcessed; i++)//遍历所有点 直到找满100个点
					{
					
						int ind3 = cloudSortInd3[i];//值较大点的索引
						// pointOri = Outlierm->points[ind3]; 
						// XYZIToXYZINormal(&pointOri, &pointWN);
						pointWN = ScanWN->points[ind3];
						pointWNormalAssociateToMap(&pointWN, &pointWNSel);//将该采样点转为map世界坐标系  使用的是odom的结果位姿
						//在现有地图点中找到距离采样点最近的一个点
						int numkdtreeo = kdtreeFromMap->nearestKSearch(pointWNSel, 1, pointSearchInd, pointSearchSqDis);
						if (numkdtreeo<=0)//没出现过 总能找到一个最近点
						{
							std::cerr<<"Error: No nearest point in map !"<<endl;
							
						}
						if (sqrt(pointSearchSqDis[0]) > OutlierThr)//此两点之间距离太大，认为是野点 
						{
							pointSearchInd.clear();
                			pointSearchSqDis.clear();
							continue;//跳过，下个点
						}
						
						CloudSampledFeat[2].push_back(pointWN);
						// outfeat_3 << pointf.x() << " " <<pointf.y() << " " << pointf.z()<< endl;
						numpicked++;
						pointSearchInd.clear();
                		pointSearchSqDis.clear();
						if (numpicked >= numselect)//若已经选够100点 结束循环
						{
							break;
						}	
					}

					numpicked = 0;//计数器清零
					for (int i = 0; i < numProcessed; i++)//遍历所有点 直到找满100个点
					{
						int ind4 = cloudSortInd4[i];//值较大点的索引
						// pointOri = Outlierm->points[ind4]; 
						// XYZIToXYZINormal(&pointOri, &pointWN);
						pointWN = ScanWN->points[ind4];
						pointWNormalAssociateToMap(&pointWN, &pointWNSel);//将该采样点转为map世界坐标系  使用的是odom的结果位姿
						//在现有地图点中找到距离采样点最近的一个点
						int numkdtreeo = kdtreeFromMap->nearestKSearch(pointWNSel, 1, pointSearchInd, pointSearchSqDis);
						if (numkdtreeo<=0)//没出现过 总能找到一个最近点
						{
							std::cerr<<"Error: No nearest point in map !"<<endl;
							
						}
						if (sqrt(pointSearchSqDis[0]) > OutlierThr)//此两点之间距离太大，认为是野点 
						{
							pointSearchInd.clear();
                			pointSearchSqDis.clear();
							continue;//跳过，下个点
						}
						
						CloudSampledFeat[3].push_back(pointWN);
						// outfeat_4 << pointf.x() << " " <<pointf.y() << " " << pointf.z()<< endl;
						numpicked++;
						pointSearchInd.clear();
                		pointSearchSqDis.clear();
						if (numpicked >= numselect)//若已经选够100点 结束循环
						{
							break;
						}	
					}
					
					numpicked = 0;//计数器清零
					for (int i = 0; i < numProcessed; i++)//遍历所有点 直到找满100个点
					{
						
						int ind5 = cloudSortInd5[i];//值较大点的索引
						// pointOri = Outlierm->points[ind5]; 
						// XYZIToXYZINormal(&pointOri, &pointWN);
						pointWN = ScanWN->points[ind5];
						pointWNormalAssociateToMap(&pointWN, &pointWNSel);//将该采样点转为map世界坐标系  使用的是odom的结果位姿
						//在现有地图点中找到距离采样点最近的一个点
						int numkdtreeo = kdtreeFromMap->nearestKSearch(pointWNSel, 1, pointSearchInd, pointSearchSqDis);
						if (numkdtreeo<=0)//没出现过 总能找到一个最近点
						{
							std::cerr<<"Error: No nearest point in map !"<<endl;
							
						}
						if (sqrt(pointSearchSqDis[0]) > OutlierThr)//此两点之间距离太大，认为是野点 
						{
							pointSearchInd.clear();
                			pointSearchSqDis.clear();
							continue;//跳过，下个点
						}
						
						CloudSampledFeat[4].push_back(pointWN);
						// outfeat_5 << pointf.x() << " " <<pointf.y() << " " << pointf.z()<< endl;
						numpicked++;
						pointSearchInd.clear();
                		pointSearchSqDis.clear();
						if (numpicked >= numselect)//若已经选够100点 结束循环
						{
							break;
						}	
					}
					
					numpicked = 0;//计数器清零
					for (int i = 0; i < numProcessed; i++)//遍历所有点 直到找满100个点
					{
						int ind6 = cloudSortInd6[i];//值较大点的索引
						// pointOri = Outlierm->points[ind6]; 
						// XYZIToXYZINormal(&pointOri, &pointWN);
						pointWN = ScanWN->points[ind6];
						pointWNormalAssociateToMap(&pointWN, &pointWNSel);//将该采样点转为map世界坐标系  使用的是odom的结果位姿
						//在现有地图点中找到距离采样点最近的一个点
						int numkdtreeo = kdtreeFromMap->nearestKSearch(pointWNSel, 1, pointSearchInd, pointSearchSqDis);
						if (numkdtreeo<=0)//没出现过 总能找到一个最近点
						{
							std::cerr<<"Error: No nearest point in map !"<<endl;
							
						}
						if (sqrt(pointSearchSqDis[0]) > OutlierThr)//此两点之间距离太大，认为是野点 
						{
							pointSearchInd.clear();
                			pointSearchSqDis.clear();
							continue;//跳过，下个点
						}
						
						CloudSampledFeat[5].push_back(pointWN);
						// outfeat_6 << pointf.x() << " " <<pointf.y() << " " << pointf.z()<< endl;
						numpicked++;
						pointSearchInd.clear();
                		pointSearchSqDis.clear();
						if (numpicked >= numselect)//若已经选够100点 结束循环
						{
							break;
						}	
					}
					
					numpicked = 0;//计数器清零
					for (int i = 0; i < numProcessed; i++)//遍历所有点 直到找满100个点
					{
						int ind7 = cloudSortInd7[i];//值较大点的索引
						// pointOri = Outlierm->points[ind7]; 
						// XYZIToXYZINormal(&pointOri, &pointWN);
						pointWN = ScanWN->points[ind7];
						pointWNormalAssociateToMap(&pointWN, &pointWNSel);//将该采样点转为map世界坐标系  使用的是odom的结果位姿
						//在现有地图点中找到距离采样点最近的一个点
						int numkdtreeo = kdtreeFromMap->nearestKSearch(pointWNSel, 1, pointSearchInd, pointSearchSqDis);
						if (numkdtreeo<=0)//没出现过 总能找到一个最近点
						{
							std::cerr<<"Error: No nearest point in map !"<<endl;
							
						}
						if (sqrt(pointSearchSqDis[0]) > OutlierThr)//此两点之间距离太大，认为是野点 
						{
							pointSearchInd.clear();
                			pointSearchSqDis.clear();
							continue;//跳过，下个点
						}
						
						CloudSampledFeat[6].push_back(pointWN);
						// outfeat_7 << pointf.x() << " " <<pointf.y() << " " << pointf.z()<< endl;
						numpicked++;
						pointSearchInd.clear();
                		pointSearchSqDis.clear();
						if (numpicked >= numselect)//若已经选够100点 结束循环
						{
							break;
						}	
					}
					numpicked = 0;//计数器清零
					for (int i = 0; i < numProcessed; i++)//遍历所有点 直到找满100个点
					{
						int ind8 = cloudSortInd8[i];//值较大点的索引
						// pointOri = Outlierm->points[ind8]; 
						// XYZIToXYZINormal(&pointOri, &pointWN);
						pointWN = ScanWN->points[ind8];
						pointWNormalAssociateToMap(&pointWN, &pointWNSel);//将该采样点转为map世界坐标系  使用的是odom的结果位姿
						//在现有地图点中找到距离采样点最近的一个点
						int numkdtreeo = kdtreeFromMap->nearestKSearch(pointWNSel, 1, pointSearchInd, pointSearchSqDis);
						if (numkdtreeo<=0)//没出现过 总能找到一个最近点
						{
							std::cerr<<"Error: No nearest point in map !"<<endl;
							
						}
						if (sqrt(pointSearchSqDis[0]) > OutlierThr)//此两点之间距离太大，认为是野点 
						{
							pointSearchInd.clear();
                			pointSearchSqDis.clear();
							continue;//跳过，下个点
						}
						
						CloudSampledFeat[7].push_back(pointWN);
						// outfeat_8 << pointf.x() << " " <<pointf.y() << " " << pointf.z()<< endl;
						numpicked++;
						pointSearchInd.clear();
                		pointSearchSqDis.clear();
						if (numpicked >= numselect)//若已经选够100点 结束循环
						{
							break;
						}	
					}
					numpicked = 0;//计数器清零
					for (int i = 0; i < numProcessed; i++)//遍历所有点 直到找满100个点
					{
						int ind9 = cloudSortInd9[i];//值较大点的索引
						// pointOri = Outlierm->points[ind9]; 
						// XYZIToXYZINormal(&pointOri, &pointWN);
						pointWN = ScanWN->points[ind9];
						pointWNormalAssociateToMap(&pointWN, &pointWNSel);//将该采样点转为map世界坐标系  使用的是odom的结果位姿
						//在现有地图点中找到距离采样点最近的一个点
						int numkdtreeo = kdtreeFromMap->nearestKSearch(pointWNSel, 1, pointSearchInd, pointSearchSqDis);
						if (numkdtreeo<=0)//没出现过 总能找到一个最近点
						{
							std::cerr<<"Error: No nearest point in map !"<<endl;
							
						}
						if (sqrt(pointSearchSqDis[0]) > OutlierThr)//此两点之间距离太大，认为是野点 
						{
							pointSearchInd.clear();
                			pointSearchSqDis.clear();
							continue;//跳过，下个点
						}
						
						CloudSampledFeat[8].push_back(pointWN);
						// outfeat_9 << pointf.x() << " " <<pointf.y() << " " << pointf.z()<< endl;
						numpicked++;
						pointSearchInd.clear();
                		pointSearchSqDis.clear();
						if (numpicked >= numselect)//若已经选够100点 结束循环
						{
							break;
						}	
					}

					printf("scan sampling time %f ms \n", t_scansample.toc());//采样总时间
					for (int i = 0; i < 9; i++)
					{
						*CloudSampled += CloudSampledFeat[i];//将所有列表采样点组合在一块
					}
					int numscansampled = CloudSampled->points.size();// 
					std::cout << "the size before sampled : " << numProcessed << " and the size after sampled is " << numscansampled << '\n';

					//暂时关闭
					//初始化A，b，x 使用动态大小矩阵
					Eigen::MatrixXd A(numscansampled, 6); 
					Eigen::Matrix< double, 6, 1 > x;
					Eigen::MatrixXd b(numscansampled, 1);
					Eigen::MatrixXd loss(numscansampled, 1);//目标函数loss
					
					//遍历每个采样点x，得到I_x和对应的投影点y 和法线//其实就是找每个点的对应点

					//将cloudprocess_raw所有点进行转换，并写入文件
					if (frameCount == 1 && (iterCount == 0 || iterCount == numICP-1))//这里的点类还是xyzi 没变
					{
						//先对当前帧下采样 再写入文件
						// pcl::PointCloud<PointType>::Ptr tmpDS_scan(new pcl::PointCloud<PointType>());//下采样后的点云
						// pcl::VoxelGrid<PointType> downSizeFilterscan;
						// downSizeFilterscan.setInputCloud(laserCloudFullRes);//注意这里的输入是原始数量的current scan
						// downSizeFilterscan.setLeafSize(DSRes, DSRes, DSRes);//可调参数 值越大，采样后越稀疏
						// downSizeFilterscan.filter(*tmpDS_scan);
						// int numtmpDS_ = tmpDS_scan->points.size();
						// std::cout<<"before "<<int(laserCloudFullRes->points.size())<<" after scan1 downsampled: "<<numtmpDS_<<endl;
						for (int i = 0; i < numprocess_raw; i++)
						{
							pointOri = laserCloudFullRes->points[i];
							pointAssociateToMap(&pointOri, &pointSel);
							Eigen::Vector3d pointc(pointSel.x, pointSel.y, pointSel.z);
							
							// x y z r g b  pointx 认为是红色
							outcloudscan << pointc.x() << " " <<pointc.y() << " " << pointc.z() << endl; 
						}
					}

					//记录每次优化被加入残差项的点的个数 即实际参与优化的点数，就是A的数据长度
					int numResidual = 0;
					double sumerror = 0;
					for (int i = 0; i < numscansampled; i++)
					{
						pointWN = CloudSampled->points[i];
						pointWNormalAssociateToMap(&pointWN, &pointWNSel);//将该特征点进行变换  使用的是当前？位姿 这步转换是否需要？如何同步地更新优化后的q_w_curr
						//在现有地图点中找到距离采样点不大于0.20m 的点
						int numkdtree = kdtreeFromMap->radiusSearch(pointWNSel, RadThr, pointSearchInd, pointSearchSqDis);
						if (numkdtree <= 0)//
						{
							// std::cout<<"WARNING: 0 BOX point found!! skip this point"<<endl;
							// std::cout<<pointSearchInd[0]<<endl;
							continue;
						}
						int numBox = pointSearchInd.size();//得到的指定半径内近邻个数
						std::vector<double> Wx;//权值数组
						std::vector<double> Ixi;//分子中的一项
						//pointsel 
						Eigen::Vector3d pointx(pointWNSel.x, pointWNSel.y, pointWNSel.z);
						//坐标变换前的点
						Eigen::Vector3d x_i(pointWN.x, pointWN.y, pointWN.z);

						//最近点
						pointxyzinormal nearestp = laserCloudFromMap->points[ pointSearchInd[0] ];
						//最近点的法线直接从数据结构中拿出
						Eigen::Vector3d nj( nearestp.normal[0], nearestp.normal[1], nearestp.normal[2] );
						Eigen::Vector3d nearestv(nearestp.x, nearestp.y, nearestp.z);
						for (int j = 0; j < numBox; j++) //只计算那个最近点 这时就退化为经典的point-planeICP numBox
						{	//当前来自地图中的点p_
							Eigen::Vector3d pcurrbox(laserCloudFromMap->points[ pointSearchInd[j] ].x,
													laserCloudFromMap->points[ pointSearchInd[j] ].y,
													laserCloudFromMap->points[ pointSearchInd[j] ].z);
							//当前来自地图中的点p_的法线
							Eigen::Vector3d normpcurr = nj; //近似
							// Eigen::Vector3d normpcurr(laserCloudFromMap->points[ pointSearchInd[j] ].normal[0],
							// 							laserCloudFromMap->points[ pointSearchInd[j] ].normal[1],
							// 							laserCloudFromMap->points[ pointSearchInd[j] ].normal[2]);
							//当前点对应的权值
							double w_j = exp(-pointSearchSqDis[j]/(h_imls*h_imls));
							//取绝对值了 不应该取绝对值
							double tmp1 = w_j*((pointx-pcurrbox).dot(normpcurr));
							// double tmp1 = abs( w_j*((pointx-pcurrbox).dot(normpcurr)) );//某一项会有nan？
							Wx.push_back(w_j);
							Ixi.push_back(tmp1);
						}

						//计算采样点x到map点隐式平面的距离
						double fenzi = std::accumulate(Ixi.begin(), Ixi.end(), 0.000001);//出现了负值？合理吗 0.000001
						double fenmu = std::accumulate(Wx.begin(), Wx.end(), 0.000001);
						double I_xi = fenzi/fenmu;//会出现NaN
						
						//x_i对应的点y_i nearestv
						Eigen::Vector3d y_i = pointx - I_xi * nj;
						// Eigen::Vector3d y_i = nearestv; //直接最近点就是对应点

						// ceres::CostFunction *cost_function = LidarPoint2PlaneICP::Create(x_i, y_i, nj); //本质上优化的是I_xi
						// problem.AddResidualBlock(cost_function, loss_function, parameters, parameters + 4);
						sumerror = sumerror + nj.dot(y_i-pointx) * nj.dot(y_i-pointx);
						// sumerror = sumerror + I_xi;
						
						//分别对A，b 赋值
						b(numResidual, 0) = nj.dot(y_i-pointx);//x_i 替换为 转换位姿后的点 pointx
						
						A(numResidual, 0) = nj.z() * pointx.y() - nj.y() * pointx.z();
						A(numResidual, 1) = nj.x() * pointx.z() - nj.z() * pointx.x();
						A(numResidual, 2) = nj.y() * pointx.x() - nj.x() * pointx.y();
						A(numResidual, 3) = nj.x();
						A(numResidual, 4) = nj.y();
						A(numResidual, 5) = nj.z();
						
						numResidual = numResidual + 1;
						//clear kdtree return vector
						pointSearchInd.clear();
						pointSearchSqDis.clear();
						/*
						//把pointx,y-i保存为txt文件 用于可视化ICP迭代过程
						if (frameCount == 1)
						{
							// x y z r g b  pointx 认为是红色
							outfile << pointx.x() << " " <<pointx.y() << " " << pointx.z() << " "
											<< (int)255 << " " << (int)0 << " " << " " << (int)0 << endl; 
							// x y z r g b  y_i 认为是绿色
							outfile << y_i.x() << " " <<y_i.y() << " " << y_i.z() << " "
											<< (int)0 << " " << (int)255 << " " << " " << (int)0 << endl;
						}
						*/
						
					}
					//一次迭代完成后，关闭文件
					if (frameCount == 1 && (iterCount == 0 || iterCount == numICP-1))
					{
						// outfile.close();
						outcloudscan.close(); 
					}
	
					printf("%d feature points are added to ResidualBlock @ %d th Iteration solver \n", numResidual, iterCount);
					
					A.conservativeResize(numResidual, NoChange);//不会改变元素！
					b.conservativeResize(numResidual, NoChange);
					loss.conservativeResize(numResidual, NoChange);
					if( ( int(A.rows()) != numResidual ) || ( int(A.cols()) != 6 ) || ( int(b.rows()) != numResidual ) || ( int(b.cols()) != 1 ) )
					{
						std::cout<<"Shape ERROR !"<<endl;
						
					}
					// std::cout<<"size of A: "<<int(A.rows())<<", "<<int(A.cols())<<endl;
					// std::cout<<"size of b: "<<int(b.rows())<<", "<<int(b.cols())<<endl;
					
					std::cout << "The sum ERROR/numpoint value is: " << sumerror/numResidual << endl;//观察变化趋势
					// std::cout << "The sum I_x value is: " << sumerror << endl;//观察变化趋势
					/*
					//求解优化
					TicToc t_solver;
					ceres::Solver::Options options;
					options.linear_solver_type = ceres::DENSE_QR;//属于列文伯格-马夸尔特方法
					options.max_num_iterations = maxnumiter1ICP;//一次优化的最大迭代次数
					options.minimizer_progress_to_stdout = false;//输出到cout 
					options.check_gradients = false;//开了检查梯度，发现我的优化有问题 应该是目前问题所在！
					options.gradient_check_relative_precision = 1e02;//1e-4是否太苛刻  好像是这个原因
					ceres::Solver::Summary summary;
					ceres::Solve(options, &problem, &summary);//开始优化
					//输出报告 debug
					std::cout<< summary.BriefReport() <<endl;
					*/
					
					//SVD求解线性最小二乘问题
					x = A.bdcSvd(ComputeThinU | ComputeThinV).solve(b);//得到了(roll,pitch,yaw,x,y,z)
					// loss = A * x - b;
					// double loss_norm = loss.norm();
					// std::cout << "The least-squares solution is:" << x.transpose() << endl;
					// std::cout << "|A*x-b|^2 is: " << loss_norm << endl;//观察
					
					
					
					//转化为四元数和位移
					double rollrad = x(0, 0);
					double pitchrad = x(1, 0);
					double yawrad = x(2, 0);
					double t_x = x(3, 0);
					double t_y = x(4, 0);
					double t_z = x(5, 0);
					//欧拉角 2 旋转矩阵
					Eigen::Matrix3d R3 = Eigen::Matrix3d::Identity();

					R3(0, 0) = cos(yawrad) * cos(pitchrad);
					R3(0, 1) = -sin(yawrad) * cos(rollrad) + cos(yawrad) * sin(pitchrad) * sin(rollrad);
					R3(0, 2) = sin(yawrad) * sin(rollrad) + cos(yawrad) * sin(pitchrad) * cos(rollrad);
					R3(1, 0) = sin(yawrad) * cos(pitchrad);
					R3(1, 1) = cos(yawrad) * cos(rollrad) + sin(yawrad) * sin(pitchrad) * sin(rollrad);
					R3(1, 2) = -cos(yawrad) * sin(rollrad) + sin(yawrad) * sin(pitchrad) * cos(rollrad);
					R3(2, 0) = -sin(pitchrad);
					R3(2, 1) = cos(pitchrad) * sin(rollrad);
					R3(2, 2) = cos(pitchrad) * cos(rollrad);
					
					//旋转矩阵转四元数
					Eigen::Quaterniond q_opt(R3);
					Eigen::Vector3d t_opt(t_x, t_y, t_z);
					// Eigen::Vector4d qopt_v(q_opt.coeffs().transpose());
					// std::cout<<"ls solution q_opt= "<<q_opt.coeffs().transpose()<<" t_opt= "<<t_opt.transpose()<<endl;

					//设置优化终止条件判断
					// std::cout<<"\n"<<iterCount<<": L2 norm of t_opt: " << t_opt.norm() <<" norm of q_opt: " <<qopt_v.norm() <<endl;

					//递增式更新！
					q_w_curr = q_opt * q_w_curr;
					t_w_curr = q_opt * t_w_curr + t_opt;
					
					
					// printf("the %d mapping solver time %f ms \n",iterCount , t_solver.toc());
					
					//输出一次mapping优化得到的位姿 w x y z 当前帧相对于map world的变换 
					// printf("\nresult q %f %f %f %f result t %f %f %f\n", parameters[3], parameters[0], parameters[1], parameters[2],
					// 	   parameters[4], parameters[5], parameters[6]);

					std::cout<<"\n"<<iterCount<<": result q= "<<q_w_curr.coeffs().transpose()<<"  result t= "<<t_w_curr.transpose()<<endl;
					
	
				}
				
				printf("\nthe frame %d mapping optimization time %f \n", frameCount, t_opt.toc());
				//20次优化后的该帧位姿最后结果
				std::cout<<"the final result of frame"<< frameCount <<": q_w_curr= "<< q_w_curr.coeffs().transpose() <<" t_w_curr= "<< t_w_curr.transpose() <<"\n"<<endl;
			}
			else//点太少 一般是第0帧
			{
				ROS_WARN("Current map model points num are not enough, skip Optimization !");
			}
			
			//迭代优化结束 更新相关的转移矩阵 选择是否更新
			// transformUpdate(); //更新了odo world 相对于 map world的变换
			// std::cout<<"the 'odo world to map world' pose of frame"<<frameCount<<": q= "<<q_wmap_wodom.coeffs().transpose()<<" t= "<<t_wmap_wodom.transpose()<<"\n"<<endl;

			TicToc t_add;
			//将当前帧的点加入到modelpoint 中 相应位置
			if (frameCount<mapsize)//说明当前model point 还没存满 直接添加
			{
				for (int i = 0; i < numProcessed; i++)//把当前优化过的scan所有点注册到地图数组指针中
				{	//将该点转移到世界坐标系下
					// pointOri = ScanWN->points[i];
					pointWNormalAssociateToMap(&ScanWN->points[i], &pointWNSel);
					ModelPointCloud[frameCount].push_back(pointWNSel);
				}

				if(int(ModelPointCloud[frameCount].points.size()) != numProcessed)
				{
					std::cout<<"ERROR when add point to modelpointcloud[ "<<frameCount<<" ] ! "<<endl;
				}


			}
			else//当前model point数组已填满100帧 去除第一个，从后面添加新的
			{
				for (int j = 0; j < mapsize-1; j++)
				{
					pcl::PointCloud<pointxyzinormal>::Ptr tmpCloud(new pcl::PointCloud<pointxyzinormal>());
					*tmpCloud = ModelPointCloud[j+1];
					int numtmpCloud = tmpCloud->points.size();
					// std::cout<<"num of ModelPointCloud["<<j+1<<"] : "<< numtmpCloud <<"\n"<<endl;
					//把数组中依次前移
					ModelPointCloud[j].clear();//->
					//应该为0
					// std::cout<<"num of ModelPointCloud["<<j<<"] after clear : "<< int(ModelPointCloud[j].points.size()) <<"\n"<<endl;
					// ModelPointCloud[j].reset(new pcl::PointCloud<PointType>());
					for (int k = 0; k < numtmpCloud; k++)
					{
						ModelPointCloud[j].push_back(tmpCloud->points[k]);
					}
					if(int(ModelPointCloud[j].points.size()) != numtmpCloud)
					{
						std::cout<<"ERROR when moving forward modelpointcloud! "<<endl;
						std::cout<<"num of ModelPointCloud["<<j<<"] : "<< int(ModelPointCloud[j].points.size())<<endl;
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
				for (int i = 0; i < numProcessed; i++)//把当前优化过的scan所有点注册到地图数组指针中
				{	//将该点转移到世界坐标系下
					pointWNormalAssociateToMap(&ScanWN->points[i], &pointWNSel);
					ModelPointCloud[mapsize-1].push_back(pointWNSel);
				}
				if(int(ModelPointCloud[mapsize-1].points.size()) != numProcessed)
				{
					std::cout<<"ERROR when add point to modelpointcloud[99]! "<<endl;
					// std::cout<<"num of ModelPointCloud["<<j<<"] : "<< int(ModelPointCloud[j].points.size) <<"\n"<<endl;
				}
				
			}
			printf("add points time %f ms\n", t_add.toc());

			TicToc t_pub;
			//发布
			sensor_msgs::PointCloud2 CloudbfPreProcess;//这里是所有预处理之前的点云
			pcl::toROSMsg(*laserCloudFullRes, CloudbfPreProcess);
			CloudbfPreProcess.header.stamp = ros::Time().fromSec(timeLaserCloudFullRes);
			CloudbfPreProcess.header.frame_id = "/camera_init"; ///camera_init
			pubCloudbfPreProcess.publish(CloudbfPreProcess);

			sensor_msgs::PointCloud2 CloudafPreProcess;//这里是所有预处理之后的点云
			pcl::toROSMsg(*CloudProcessed, CloudafPreProcess);
			CloudafPreProcess.header.stamp = ros::Time().fromSec(timeLaserCloudFullRes);
			CloudafPreProcess.header.frame_id = "/camera_init";
			pubCloudafPreProcess.publish(CloudafPreProcess);
			//发布前要变回xyzi类型吗 先不变试试  watch! 可以
			//for now 发布采样后的特征点
			sensor_msgs::PointCloud2 laserCloudSampled;//
			pcl::toROSMsg(*CloudSampled, laserCloudSampled);
			laserCloudSampled.header.stamp = ros::Time().fromSec(timeLaserCloudFullRes);
			laserCloudSampled.header.frame_id = "/camera_init";
			pubCloudSampled.publish(laserCloudSampled);

			// pub each sampled feature list
			if(PUB_EACH_List)
			{
				for(int i = 0; i< 9; i++)
				{
					sensor_msgs::PointCloud2 ListMsg;
					pcl::toROSMsg(CloudSampledFeat[i], ListMsg);
					ListMsg.header.stamp = ros::Time().fromSec(timeLaserCloudFullRes);
					ListMsg.header.frame_id = "/camera_init";
					pubEachFeatList[i].publish(ListMsg);
				}
			}
			for (int i = 0; i < 9; i++)
			{
				CloudSampledFeat[i].clear();//将所有列表采样点清零
				if(int(CloudSampledFeat[i].points.size()) != 0)
				{
					std::cout<<"Failed to clear CloudSampledFeat !\n"<<endl;
				}
			}
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
			// int laserCloudFullResNum = laserCloudFullRes->points.size();
			// for (int i = 0; i < laserCloudFullResNum; i++)
			// {
			// 	pointAssociateToMap(&laserCloudFullRes->points[i], &laserCloudFullRes->points[i]);
			// }

			// sensor_msgs::PointCloud2 laserCloudFullRes3;//当前帧的所有点  多于Cloudprocessed
			// pcl::toROSMsg(*laserCloudFullRes, laserCloudFullRes3);
			// laserCloudFullRes3.header.stamp = ros::Time().fromSec(timeLaserCloudFullRes);
			// laserCloudFullRes3.header.frame_id = "/camera_init";
			// pubLaserCloudFullRes.publish(laserCloudFullRes3);///velodyne_cloud_registered 当前帧已注册的点

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
			br.sendTransform(tf::StampedTransform(transform, odomAftMapped.header.stamp, "/camera", "/aft_mapped"));

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
	// downSizeFilterSurf.setLeafSize(planeRes, planeRes, planeRes); //用于下采样地面特征点！  也对保存的model cloud 进行下采样

	//接收来自laserodo.cpp发来的处理后的点云/cloud_4 topic 
	//暂时关闭 去除运动物体模块 直接拿来全部一帧点云
	// ros::Subscriber subCloudprocessed = nh.subscribe<sensor_msgs::PointCloud2>("/cloud_4", 100, CloudprocessedHandler);

	//这3个只有前十帧才非空
	// ros::Subscriber subLaserCloudCornerLast = nh.subscribe<sensor_msgs::PointCloud2>("/laser_cloud_corner_last", 100, laserCloudCornerLastHandler);

	// ros::Subscriber subLaserCloudSurfLast = nh.subscribe<sensor_msgs::PointCloud2>("/laser_cloud_surf_last", 100, laserCloudSurfLastHandler);

	// ros::Subscriber subLaserOdometry = nh.subscribe<nav_msgs::Odometry>("/laser_odom_to_init", 100, laserOdometryHandler);
	///velodyne_cloud_3
	ros::Subscriber subLaserCloudFullRes = nh.subscribe<sensor_msgs::PointCloud2>("/velodyne_cloud_2", 100, laserCloudFullResHandler);
	//订阅odometry部分得到的当前帧的地面点
	// ros::Subscriber subCloudGroundLast = nh.subscribe<sensor_msgs::PointCloud2>("/cloud_3_Ground", 100, CloudGroundLastHandler);
	//参数中的100 ：排队等待处理的传入消息数（超出此队列容量的消息将被丢弃）

	pubCloudbfPreProcess = nh.advertise<sensor_msgs::PointCloud2>("/cloud_before_preprocess", 100);//发布当前采样前的点 和cloud4一样
	pubCloudafPreProcess = nh.advertise<sensor_msgs::PointCloud2>("/cloud_after_preprocess", 100); //体素滤波之后
 
	pubCloudSampled = nh.advertise<sensor_msgs::PointCloud2>("/laser_cloud_sampled", 100);//发布当前采样后的点

	// pubLaserCloudSurround = nh.advertise<sensor_msgs::PointCloud2>("/laser_cloud_surround", 100);//周围的地图点

	// pubLaserCloudMap = nh.advertise<sensor_msgs::PointCloud2>("/laser_cloud_map", 100);//更多的地图点

	// pubLaserCloudFullRes = nh.advertise<sensor_msgs::PointCloud2>("/velodyne_cloud_registered", 100);//当前帧（已注册）的所有点

	pubOdomAftMapped = nh.advertise<nav_msgs::Odometry>("/aft_mapped_to_init", 100);//优化后的位姿？

	//pubOdomAftMappedHighFrec = nh.advertise<nav_msgs::Odometry>("/aft_mapped_to_init_high_frec", 100);//接收odo的位姿 高频输出 不是最后的优化结果

	pubLaserAfterMappedPath = nh.advertise<nav_msgs::Path>("/aft_mapped_path", 100);

	// for (int i = 0; i < laserCloudNum; i++)
	// {
	// 	laserCloudCornerArray[i].reset(new pcl::PointCloud<PointType>());
	// 	laserCloudSurfArray[i].reset(new pcl::PointCloud<PointType>());
	// }

	if(PUB_EACH_List)//默认false
    {
        for(int i = 0; i < 9; i++) //
        {
            ros::Publisher tmp = nh.advertise<sensor_msgs::PointCloud2>("/feature_listid_" + std::to_string(i), 100);
            pubEachFeatList.push_back(tmp);
        }
    }

	//mapping过程 单开一个线程
	std::thread mapping_process{imls};

	ros::spin();

	return 0;
}