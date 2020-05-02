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

#include <cmath>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <geometry_msgs/PoseStamped.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl_conversions/pcl_conversions.h>

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
#include <pcl/kdtree/kdtree.h>//kd树搜索算法
#include <pcl/segmentation/extract_clusters.h>//欧式聚类分割

#include <ros/ros.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/PointCloud2.h>
#include <tf/transform_datatypes.h>
#include <tf/transform_broadcaster.h>
#include <eigen3/Eigen/Dense>
#include <mutex>
#include <queue>

#include "aloam_velodyne/common.h"
#include "aloam_velodyne/tic_toc.h"
#include "lidarFactor.hpp"
using namespace std;

#define DISTORTION 0 //仅对于kitti distortion-free
#define clusterRMThs 100 //小于边界框且点数大于该阈值的 就被去除
// //是否是地面点，0表示是地面点
#define GroundThs -1.65 //z轴小于等于-1.70的点，暂时认为是地面点 可调节
//small group removal 界限
#define Xbox 14
#define Ybox 14
#define Zbox 4

int corner_correspondence = 0, plane_correspondence = 0;
//一个scan的时间周期，10HZ
constexpr double SCAN_PERIOD = 0.1;
//？
constexpr double DISTANCE_SQ_THRESHOLD = 25;
constexpr double NEARBY_SCAN = 2.5;

//跳帧数，控制发给laserMapping的频率
int skipFrameNum = 1;
bool systemInited = false;
bool systemInited_odom = false;
const int systemDelay_odom = 0;//只有前10帧计算odometry的位姿 4662
int systemInitCount_odom = 0;// new:在laserOdometry中计数当前处理的帧序号 main 中有framecount了，作用一样！

//时间戳信息
double timeCornerPointsSharp = 0;
double timeCornerPointsLessSharp = 0;
double timeSurfPointsFlat = 0;
double timeSurfPointsLessFlat = 0;
double timeLaserCloudFullRes = 0;

//flann? 上一帧的lesssharp点、less flat点构成的kdtree
pcl::KdTreeFLANN<pcl::PointXYZI>::Ptr kdtreeCornerLast(new pcl::KdTreeFLANN<pcl::PointXYZI>());
pcl::KdTreeFLANN<pcl::PointXYZI>::Ptr kdtreeSurfLast(new pcl::KdTreeFLANN<pcl::PointXYZI>());

//接收scanRegistration传来的四类特征
pcl::PointCloud<PointType>::Ptr cornerPointsSharp(new pcl::PointCloud<PointType>());
pcl::PointCloud<PointType>::Ptr cornerPointsLessSharp(new pcl::PointCloud<PointType>());
pcl::PointCloud<PointType>::Ptr surfPointsFlat(new pcl::PointCloud<PointType>());
pcl::PointCloud<PointType>::Ptr surfPointsLessFlat(new pcl::PointCloud<PointType>());

//接收上一帧的特征点
pcl::PointCloud<PointType>::Ptr laserCloudCornerLast(new pcl::PointCloud<PointType>());
pcl::PointCloud<PointType>::Ptr laserCloudSurfLast(new pcl::PointCloud<PointType>());
//接收传来的所有点
pcl::PointCloud<PointType>::Ptr laserCloudFullRes(new pcl::PointCloud<PointType>());

//数量？
int laserCloudCornerLastNum = 0;
int laserCloudSurfLastNum = 0;

// Transformation from current frame to world frame 当前帧相对于世界坐标的变换
Eigen::Quaterniond q_w_curr(1, 0, 0, 0);//该值只用于了初始化，即第一帧就是世界坐标
Eigen::Vector3d t_w_curr(0, 0, 0);

// q_last_curr(x, y, z, w), t_last_curr
double para_q[4] = {0, 0, 0, 1};
double para_t[3] = {0, 0, 0};
//当前帧to 上一阵的变换
Eigen::Map<Eigen::Quaterniond> q_last_curr(para_q);
Eigen::Map<Eigen::Vector3d> t_last_curr(para_t);

//？
std::queue<sensor_msgs::PointCloud2ConstPtr> cornerSharpBuf;
std::queue<sensor_msgs::PointCloud2ConstPtr> cornerLessSharpBuf;
std::queue<sensor_msgs::PointCloud2ConstPtr> surfFlatBuf;
std::queue<sensor_msgs::PointCloud2ConstPtr> surfLessFlatBuf;
std::queue<sensor_msgs::PointCloud2ConstPtr> fullPointsBuf;//存储全部点
std::mutex mBuf;

// undistort lidar point 把一个sweep内的点转为该帧初始点坐标系下 q_last_curr怎么得到的？
void TransformToStart(PointType const *const pi, PointType *const po)
{
    //interpolation ratio 插值系数
    double s;
    if (DISTORTION) //0
        s = (pi->intensity - int(pi->intensity)) / SCAN_PERIOD;
    else
        s = 1.0;
    //s = 1;
    //在单位旋转和q_last2curr之间插值，也就是对q_last2curr乘一个系数 1的话就等于q_last_curr 因为kitti数据无distort，都已经配准在帧末坐标系了
    //q_last_curr怎么得到的？
    Eigen::Quaterniond q_point_last = Eigen::Quaterniond::Identity().slerp(s, q_last_curr);
    Eigen::Vector3d t_point_last = s * t_last_curr;//得到了当前点（某一sweep之内）相对上一阵的变换 任意点的相对（上一帧）位姿
    Eigen::Vector3d point(pi->x, pi->y, pi->z);
    Eigen::Vector3d un_point = q_point_last * point + t_point_last;//转化为上一帧末，即当前帧初始的坐标系

    po->x = un_point.x();
    po->y = un_point.y();
    po->z = un_point.z();
    po->intensity = pi->intensity;
}

// transform all lidar points to the start of the next frame
//IMLSSLAM中scan egomotion?? 把一个sweep中的点转为end of scan 的坐标系
void TransformToEnd(PointType const *const pi, PointType *const po)
{
    // undistort point first
    pcl::PointXYZI un_point_tmp;
    TransformToStart(pi, &un_point_tmp);

    Eigen::Vector3d un_point(un_point_tmp.x, un_point_tmp.y, un_point_tmp.z);
    Eigen::Vector3d point_end = q_last_curr.inverse() * (un_point - t_last_curr); //逆变换 相当于T_curr_last*p_last

    po->x = point_end.x();
    po->y = point_end.y();
    po->z = point_end.z();

    //Remove distortion time info
    po->intensity = int(pi->intensity);
}

/*

void laserCloudSharpHandler(const sensor_msgs::PointCloud2ConstPtr &cornerPointsSharp2)
{
    mBuf.lock();
    cornerSharpBuf.push(cornerPointsSharp2);
    mBuf.unlock();
}

void laserCloudLessSharpHandler(const sensor_msgs::PointCloud2ConstPtr &cornerPointsLessSharp2)
{
    mBuf.lock();
    cornerLessSharpBuf.push(cornerPointsLessSharp2);
    mBuf.unlock();
}

void laserCloudFlatHandler(const sensor_msgs::PointCloud2ConstPtr &surfPointsFlat2)
{
    mBuf.lock();
    surfFlatBuf.push(surfPointsFlat2);
    mBuf.unlock();
}

void laserCloudLessFlatHandler(const sensor_msgs::PointCloud2ConstPtr &surfPointsLessFlat2)
{
    mBuf.lock();
    surfLessFlatBuf.push(surfPointsLessFlat2);
    mBuf.unlock();
}
*/


//receive all point cloud 来自scanRegistration
void laserCloudFullResHandler(const sensor_msgs::PointCloud2ConstPtr &laserCloudFullRes2)
{
    mBuf.lock();
    fullPointsBuf.push(laserCloudFullRes2);
    mBuf.unlock();
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "laserOdometry");
    ros::NodeHandle nh;
    //跳帧数，控制发给mapping的频率 1
    nh.param<int>("mapping_skip_frame", skipFrameNum, 2);

    printf("Mapping %d Hz \n", 10 / skipFrameNum);//仍是10hz？

    //订阅scanRegistration输出的5各节点，接收输出过来的消息，调用对应的函数去处理
    //*********
    //以下四类点只有在前10帧才非空
    // ros::Subscriber subCornerPointsSharp = nh.subscribe<sensor_msgs::PointCloud2>("/laser_cloud_sharp", 100, laserCloudSharpHandler);

    // ros::Subscriber subCornerPointsLessSharp = nh.subscribe<sensor_msgs::PointCloud2>("/laser_cloud_less_sharp", 100, laserCloudLessSharpHandler);

    // ros::Subscriber subSurfPointsFlat = nh.subscribe<sensor_msgs::PointCloud2>("/laser_cloud_flat", 100, laserCloudFlatHandler);

    // ros::Subscriber subSurfPointsLessFlat = nh.subscribe<sensor_msgs::PointCloud2>("/laser_cloud_less_flat", 100, laserCloudLessFlatHandler);

    /************/
    ros::Subscriber subLaserCloudFullRes = nh.subscribe<sensor_msgs::PointCloud2>("/velodyne_cloud_2", 100, laserCloudFullResHandler);

    //在对应topic发布消息
    //这两个只在前10帧才发送 吧
    // ros::Publisher pubLaserCloudCornerLast = nh.advertise<sensor_msgs::PointCloud2>("/laser_cloud_corner_last", 100);
    // ros::Publisher pubLaserCloudSurfLast = nh.advertise<sensor_msgs::PointCloud2>("/laser_cloud_surf_last", 100);
    //debug 这个被后面暂时用作 “cloud_4”
    ros::Publisher pubLaserCloudFullRes = nh.advertise<sensor_msgs::PointCloud2>("/velodyne_cloud_3", 100);//输出新的点云，其实和cloud2一样

    // ros::Publisher publaserCloudWOGround = nh.advertise<sensor_msgs::PointCloud2>("/cloud_3_woGround", 100);
    // ros::Publisher publaserCloudGround = nh.advertise<sensor_msgs::PointCloud2>("/cloud_3_Ground", 100);
    // ros::Publisher pubCloudObjectRemoval = nh.advertise<sensor_msgs::PointCloud2>("/cloud_3_objectremoval", 100);//去除小物体后的点云（不含地面）
    //debug 这个为空
    ros::Publisher pubObjectRemovalGround = nh.advertise<sensor_msgs::PointCloud2>("/cloud_4", 100);//去除小物体并添加了地面 传给lasermapping.cpp.0

    //*****  只在前10帧 非空
    // ros::Publisher pubLaserOdometry = nh.advertise<nav_msgs::Odometry>("/laser_odom_to_init", 100);//输出odo T_WV

    // ros::Publisher pubLaserPath = nh.advertise<nav_msgs::Path>("/laser_odom_path", 100);

    // nav_msgs::Path laserPath;//以上三者有区别吗?
    /******/

    int frameCount = 0;
    ros::Rate rate(100);//

    while (ros::ok())
    {
        // std::cout<<fullPointsBuf.empty()<<endl;
        ros::spinOnce();

        // std::cout<<fullPointsBuf.empty()<<endl;
        if (!fullPointsBuf.empty())
        {//当各类点非空，确保接收到才开始
            std::cout<<"points not empty!"<<endl;
            //时间
            timeLaserCloudFullRes = fullPointsBuf.front()->header.stamp.toSec(); //只有这个点云是非空的

            //用点云数据容器接收Buf中各类点云
            
            mBuf.lock();

            laserCloudFullRes->clear();
            pcl::fromROSMsg(*fullPointsBuf.front(), *laserCloudFullRes);
            fullPointsBuf.pop();
            mBuf.unlock();

            TicToc t_whole;
            TicToc t_ego;
            // initializing
            if (!systemInited)//只有初始帧走这里
            {
                systemInited = true;
                std::cout << "LaserOdometry: Initialization finished \n";
            }//第一帧也不用undistort

            TicToc t_pub;
            pcl::PointCloud<PointType>::Ptr cloud_cluster_allWGround (new pcl::PointCloud<PointType>());
            *cloud_cluster_allWGround = *laserCloudFullRes;
            
            //每隔skip帧才发给Mapping部分 这里还是1 10hz
            if (frameCount % skipFrameNum == 0)
            {
                frameCount = 0;
                
                sensor_msgs::PointCloud2 laserCloudFullRes3;
                pcl::toROSMsg(*laserCloudFullRes, laserCloudFullRes3);
                laserCloudFullRes3.header.stamp = ros::Time().fromSec(timeLaserCloudFullRes);
                laserCloudFullRes3.header.frame_id = "/camera";
                pubLaserCloudFullRes.publish(laserCloudFullRes3);
                /*
                sensor_msgs::PointCloud2 laserCloudWOGroundMsg;
                pcl::toROSMsg(*laserCloudWOGround, laserCloudWOGroundMsg);
                laserCloudWOGroundMsg.header.stamp = ros::Time().fromSec(timeLaserCloudFullRes);
                laserCloudWOGroundMsg.header.frame_id = "/camera";
                publaserCloudWOGround.publish(laserCloudWOGroundMsg);

                sensor_msgs::PointCloud2 laserCloudGroundMsg;
                pcl::toROSMsg(*laserCloudGround, laserCloudGroundMsg);
                laserCloudGroundMsg.header.stamp = ros::Time().fromSec(timeLaserCloudFullRes);
                laserCloudGroundMsg.header.frame_id = "/camera";
                publaserCloudGround.publish(laserCloudGroundMsg);

                sensor_msgs::PointCloud2 laserCloudObjectRemovalMsg;
                pcl::toROSMsg(*cloud_cluster_all, laserCloudObjectRemovalMsg);
                laserCloudObjectRemovalMsg.header.stamp = ros::Time().fromSec(timeLaserCloudFullRes);
                laserCloudObjectRemovalMsg.header.frame_id = "/camera";
                pubCloudObjectRemoval.publish(laserCloudObjectRemovalMsg);
                */
                sensor_msgs::PointCloud2 laserObjectRemovalGroundMsg;
                pcl::toROSMsg(*cloud_cluster_allWGround, laserObjectRemovalGroundMsg);
                laserObjectRemovalGroundMsg.header.stamp = ros::Time().fromSec(timeLaserCloudFullRes);
                laserObjectRemovalGroundMsg.header.frame_id = "/camera";
                pubObjectRemovalGround.publish(laserObjectRemovalGroundMsg);//发布后，lasermapping.cpp
            }
            printf("publication time %f ms \n", t_pub.toc());
            printf("whole laserOdometry time %f ms *****************************\n \n", t_whole.toc());
            if(t_whole.toc() > 100)
                ROS_WARN("odometry process over 100ms");

            frameCount++;
            systemInitCount_odom++;//!
        }

        rate.sleep();
    }
    return 0;
}