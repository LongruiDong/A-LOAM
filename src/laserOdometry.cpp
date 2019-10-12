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

#define DISTORTION 0 //仅对于kitti distortion-free

// //是否是地面点，0表示是地面点
// int GroundptsFlag[400000];
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
int skipFrameNum = 5;
bool systemInited = false;

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
// //去除地面后的点
// pcl::PointCloud<PointType>::Ptr laserCloudWOGround(new pcl::PointCloud<PointType>());
// //地面点
// pcl::PointCloud<PointType>::Ptr laserCloudGround(new pcl::PointCloud<PointType>());

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
    //q_last_curr怎么得到的？
    Eigen::Quaterniond q_point_last = Eigen::Quaterniond::Identity().slerp(s, q_last_curr);//在单位旋转和q_last2curr之间插值，也就是对q_last2curr乘一个系数
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


//*****
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

/******/

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
    ros::Subscriber subCornerPointsSharp = nh.subscribe<sensor_msgs::PointCloud2>("/laser_cloud_sharp", 100, laserCloudSharpHandler);

    ros::Subscriber subCornerPointsLessSharp = nh.subscribe<sensor_msgs::PointCloud2>("/laser_cloud_less_sharp", 100, laserCloudLessSharpHandler);

    ros::Subscriber subSurfPointsFlat = nh.subscribe<sensor_msgs::PointCloud2>("/laser_cloud_flat", 100, laserCloudFlatHandler);

    ros::Subscriber subSurfPointsLessFlat = nh.subscribe<sensor_msgs::PointCloud2>("/laser_cloud_less_flat", 100, laserCloudLessFlatHandler);

    /************/
    ros::Subscriber subLaserCloudFullRes = nh.subscribe<sensor_msgs::PointCloud2>("/velodyne_cloud_2", 100, laserCloudFullResHandler);

    //在对应topic发布消息

    ros::Publisher pubLaserCloudCornerLast = nh.advertise<sensor_msgs::PointCloud2>("/laser_cloud_corner_last", 100);
    ros::Publisher pubLaserCloudSurfLast = nh.advertise<sensor_msgs::PointCloud2>("/laser_cloud_surf_last", 100);

    ros::Publisher pubLaserCloudFullRes = nh.advertise<sensor_msgs::PointCloud2>("/velodyne_cloud_3", 100);//输出新的点云

    ros::Publisher publaserCloudWOGround = nh.advertise<sensor_msgs::PointCloud2>("/cloud_3_woGround", 100);
    ros::Publisher publaserCloudGround = nh.advertise<sensor_msgs::PointCloud2>("/cloud_3_Ground", 100);
    ros::Publisher pubCloudObjectRemoval = nh.advertise<sensor_msgs::PointCloud2>("/cloud_3_objectremoval", 100);//去除小物体后的点云（不含地面）

    ros::Publisher pubObjectRemovalGround = nh.advertise<sensor_msgs::PointCloud2>("/cloud_4", 100);//去除小物体并添加了地面 传给lasermapping.cpp.0

    //*****
    ros::Publisher pubLaserOdometry = nh.advertise<nav_msgs::Odometry>("/laser_odom_to_init", 100);//输出odo T_WV

    ros::Publisher pubLaserPath = nh.advertise<nav_msgs::Path>("/laser_odom_path", 100);

    nav_msgs::Path laserPath;//以上三者有区别吗?
    /******/

    int frameCount = 0;
    ros::Rate rate(100);//

    while (ros::ok())
    {
        ros::spinOnce();

        if (!cornerSharpBuf.empty() && !cornerLessSharpBuf.empty() &&
            !surfFlatBuf.empty() && !surfLessFlatBuf.empty() &&
            !fullPointsBuf.empty())
        //if (!fullPointsBuf.empty())
        {//当各类点非空，确保接收到才开始
            //时间
            timeCornerPointsSharp = cornerSharpBuf.front()->header.stamp.toSec();
            timeCornerPointsLessSharp = cornerLessSharpBuf.front()->header.stamp.toSec();
            timeSurfPointsFlat = surfFlatBuf.front()->header.stamp.toSec();
            timeSurfPointsLessFlat = surfLessFlatBuf.front()->header.stamp.toSec();
            timeLaserCloudFullRes = fullPointsBuf.front()->header.stamp.toSec();

            if (timeCornerPointsSharp != timeLaserCloudFullRes ||
                timeCornerPointsLessSharp != timeLaserCloudFullRes ||
                timeSurfPointsFlat != timeLaserCloudFullRes ||
                timeSurfPointsLessFlat != timeLaserCloudFullRes)
            {
                printf("unsync messeage!");//各类点应时间同步
                ROS_BREAK();
            }

            //用点云数据容器接收Buf中各类点云
            mBuf.lock();
            cornerPointsSharp->clear();
            pcl::fromROSMsg(*cornerSharpBuf.front(), *cornerPointsSharp);
            cornerSharpBuf.pop();

            cornerPointsLessSharp->clear();
            pcl::fromROSMsg(*cornerLessSharpBuf.front(), *cornerPointsLessSharp);
            cornerLessSharpBuf.pop();

            surfPointsFlat->clear();
            pcl::fromROSMsg(*surfFlatBuf.front(), *surfPointsFlat);
            surfFlatBuf.pop();

            surfPointsLessFlat->clear();
            pcl::fromROSMsg(*surfLessFlatBuf.front(), *surfPointsLessFlat);
            surfLessFlatBuf.pop();

            laserCloudFullRes->clear();
            pcl::fromROSMsg(*fullPointsBuf.front(), *laserCloudFullRes);
            fullPointsBuf.pop();
            mBuf.unlock();

            TicToc t_whole;
            TicToc t_ego;
            // initializing
            if (!systemInited)
            {
                systemInited = true;
                std::cout << "Initialization finished \n";
            }//第一帧也不用undistort
            else
            {//从第二帧进行以下步骤
                int cornerPointsSharpNum = cornerPointsSharp->points.size();
                int surfPointsFlatNum = surfPointsFlat->points.size();

                TicToc t_opt;
                for (size_t opti_counter = 0; opti_counter < 2; ++opti_counter)//优化两次？
                {
                    //特征点的相关性
                    corner_correspondence = 0;
                    plane_correspondence = 0;

                    //ceres::LossFunction *loss_function = NULL;
                    ceres::LossFunction *loss_function = new ceres::HuberLoss(0.1);
                    ceres::LocalParameterization *q_parameterization =
                        new ceres::EigenQuaternionParameterization();
                    ceres::Problem::Options problem_options;

                    ceres::Problem problem(problem_options);
                    problem.AddParameterBlock(para_q, 4, q_parameterization);
                    problem.AddParameterBlock(para_t, 3);

                    pcl::PointXYZI pointSel;
                    std::vector<int> pointSearchInd;
                    std::vector<float> pointSearchSqDis;

                    TicToc t_data;
                    // find correspondence for corner features
                    for (int i = 0; i < cornerPointsSharpNum; ++i)
                    {
                        TransformToStart(&(cornerPointsSharp->points[i]), &pointSel);
                        kdtreeCornerLast->nearestKSearch(pointSel, 1, pointSearchInd, pointSearchSqDis);

                        int closestPointInd = -1, minPointInd2 = -1;
                        if (pointSearchSqDis[0] < DISTANCE_SQ_THRESHOLD)
                        {
                            closestPointInd = pointSearchInd[0];
                            int closestPointScanID = int(laserCloudCornerLast->points[closestPointInd].intensity);

                            double minPointSqDis2 = DISTANCE_SQ_THRESHOLD;
                            // search in the direction of increasing scan line
                            for (int j = closestPointInd + 1; j < (int)laserCloudCornerLast->points.size(); ++j)
                            {
                                // if in the same scan line, continue
                                if (int(laserCloudCornerLast->points[j].intensity) <= closestPointScanID)
                                    continue;

                                // if not in nearby scans, end the loop
                                if (int(laserCloudCornerLast->points[j].intensity) > (closestPointScanID + NEARBY_SCAN))
                                    break;

                                double pointSqDis = (laserCloudCornerLast->points[j].x - pointSel.x) *
                                                        (laserCloudCornerLast->points[j].x - pointSel.x) +
                                                    (laserCloudCornerLast->points[j].y - pointSel.y) *
                                                        (laserCloudCornerLast->points[j].y - pointSel.y) +
                                                    (laserCloudCornerLast->points[j].z - pointSel.z) *
                                                        (laserCloudCornerLast->points[j].z - pointSel.z);

                                if (pointSqDis < minPointSqDis2)
                                {
                                    // find nearer point
                                    minPointSqDis2 = pointSqDis;
                                    minPointInd2 = j;
                                }
                            }

                            // search in the direction of decreasing scan line
                            for (int j = closestPointInd - 1; j >= 0; --j)
                            {
                                // if in the same scan line, continue
                                if (int(laserCloudCornerLast->points[j].intensity) >= closestPointScanID)
                                    continue;

                                // if not in nearby scans, end the loop
                                if (int(laserCloudCornerLast->points[j].intensity) < (closestPointScanID - NEARBY_SCAN))
                                    break;

                                double pointSqDis = (laserCloudCornerLast->points[j].x - pointSel.x) *
                                                        (laserCloudCornerLast->points[j].x - pointSel.x) +
                                                    (laserCloudCornerLast->points[j].y - pointSel.y) *
                                                        (laserCloudCornerLast->points[j].y - pointSel.y) +
                                                    (laserCloudCornerLast->points[j].z - pointSel.z) *
                                                        (laserCloudCornerLast->points[j].z - pointSel.z);

                                if (pointSqDis < minPointSqDis2)
                                {
                                    // find nearer point
                                    minPointSqDis2 = pointSqDis;
                                    minPointInd2 = j;
                                }
                            }
                        }
                        if (minPointInd2 >= 0) // both closestPointInd and minPointInd2 is valid
                        {
                            Eigen::Vector3d curr_point(cornerPointsSharp->points[i].x,
                                                       cornerPointsSharp->points[i].y,
                                                       cornerPointsSharp->points[i].z);
                            Eigen::Vector3d last_point_a(laserCloudCornerLast->points[closestPointInd].x,
                                                         laserCloudCornerLast->points[closestPointInd].y,
                                                         laserCloudCornerLast->points[closestPointInd].z);
                            Eigen::Vector3d last_point_b(laserCloudCornerLast->points[minPointInd2].x,
                                                         laserCloudCornerLast->points[minPointInd2].y,
                                                         laserCloudCornerLast->points[minPointInd2].z);

                            double s;
                            if (DISTORTION)
                                s = (cornerPointsSharp->points[i].intensity - int(cornerPointsSharp->points[i].intensity)) / SCAN_PERIOD;
                            else
                                s = 1.0;
                            ceres::CostFunction *cost_function = LidarEdgeFactor::Create(curr_point, last_point_a, last_point_b, s);
                            problem.AddResidualBlock(cost_function, loss_function, para_q, para_t);
                            corner_correspondence++;
                        }
                    }

                    // find correspondence for plane features
                    for (int i = 0; i < surfPointsFlatNum; ++i)
                    {
                        TransformToStart(&(surfPointsFlat->points[i]), &pointSel);
                        kdtreeSurfLast->nearestKSearch(pointSel, 1, pointSearchInd, pointSearchSqDis);

                        int closestPointInd = -1, minPointInd2 = -1, minPointInd3 = -1;
                        if (pointSearchSqDis[0] < DISTANCE_SQ_THRESHOLD)
                        {
                            closestPointInd = pointSearchInd[0];

                            // get closest point's scan ID
                            int closestPointScanID = int(laserCloudSurfLast->points[closestPointInd].intensity);
                            double minPointSqDis2 = DISTANCE_SQ_THRESHOLD, minPointSqDis3 = DISTANCE_SQ_THRESHOLD;

                            // search in the direction of increasing scan line
                            for (int j = closestPointInd + 1; j < (int)laserCloudSurfLast->points.size(); ++j)
                            {
                                // if not in nearby scans, end the loop
                                if (int(laserCloudSurfLast->points[j].intensity) > (closestPointScanID + NEARBY_SCAN))
                                    break;

                                double pointSqDis = (laserCloudSurfLast->points[j].x - pointSel.x) *
                                                        (laserCloudSurfLast->points[j].x - pointSel.x) +
                                                    (laserCloudSurfLast->points[j].y - pointSel.y) *
                                                        (laserCloudSurfLast->points[j].y - pointSel.y) +
                                                    (laserCloudSurfLast->points[j].z - pointSel.z) *
                                                        (laserCloudSurfLast->points[j].z - pointSel.z);

                                // if in the same or lower scan line
                                if (int(laserCloudSurfLast->points[j].intensity) <= closestPointScanID && pointSqDis < minPointSqDis2)
                                {
                                    minPointSqDis2 = pointSqDis;
                                    minPointInd2 = j;
                                }
                                // if in the higher scan line
                                else if (int(laserCloudSurfLast->points[j].intensity) > closestPointScanID && pointSqDis < minPointSqDis3)
                                {
                                    minPointSqDis3 = pointSqDis;
                                    minPointInd3 = j;
                                }
                            }

                            // search in the direction of decreasing scan line
                            for (int j = closestPointInd - 1; j >= 0; --j)
                            {
                                // if not in nearby scans, end the loop
                                if (int(laserCloudSurfLast->points[j].intensity) < (closestPointScanID - NEARBY_SCAN))
                                    break;

                                double pointSqDis = (laserCloudSurfLast->points[j].x - pointSel.x) *
                                                        (laserCloudSurfLast->points[j].x - pointSel.x) +
                                                    (laserCloudSurfLast->points[j].y - pointSel.y) *
                                                        (laserCloudSurfLast->points[j].y - pointSel.y) +
                                                    (laserCloudSurfLast->points[j].z - pointSel.z) *
                                                        (laserCloudSurfLast->points[j].z - pointSel.z);

                                // if in the same or higher scan line
                                if (int(laserCloudSurfLast->points[j].intensity) >= closestPointScanID && pointSqDis < minPointSqDis2)
                                {
                                    minPointSqDis2 = pointSqDis;
                                    minPointInd2 = j;
                                }
                                else if (int(laserCloudSurfLast->points[j].intensity) < closestPointScanID && pointSqDis < minPointSqDis3)
                                {
                                    // find nearer point
                                    minPointSqDis3 = pointSqDis;
                                    minPointInd3 = j;
                                }
                            }

                            if (minPointInd2 >= 0 && minPointInd3 >= 0)
                            {

                                Eigen::Vector3d curr_point(surfPointsFlat->points[i].x,
                                                            surfPointsFlat->points[i].y,
                                                            surfPointsFlat->points[i].z);
                                Eigen::Vector3d last_point_a(laserCloudSurfLast->points[closestPointInd].x,
                                                                laserCloudSurfLast->points[closestPointInd].y,
                                                                laserCloudSurfLast->points[closestPointInd].z);
                                Eigen::Vector3d last_point_b(laserCloudSurfLast->points[minPointInd2].x,
                                                                laserCloudSurfLast->points[minPointInd2].y,
                                                                laserCloudSurfLast->points[minPointInd2].z);
                                Eigen::Vector3d last_point_c(laserCloudSurfLast->points[minPointInd3].x,
                                                                laserCloudSurfLast->points[minPointInd3].y,
                                                                laserCloudSurfLast->points[minPointInd3].z);

                                double s;
                                if (DISTORTION)
                                    s = (surfPointsFlat->points[i].intensity - int(surfPointsFlat->points[i].intensity)) / SCAN_PERIOD;
                                else
                                    s = 1.0;
                                ceres::CostFunction *cost_function = LidarPlaneFactor::Create(curr_point, last_point_a, last_point_b, last_point_c, s);
                                problem.AddResidualBlock(cost_function, loss_function, para_q, para_t);
                                plane_correspondence++;
                            }
                        }
                    }

                    //printf("coner_correspondance %d, plane_correspondence %d \n", corner_correspondence, plane_correspondence);
                    printf("data association time %f ms \n", t_data.toc());

                    if ((corner_correspondence + plane_correspondence) < 10)
                    {
                        printf("less correspondence! *************************************************\n");
                    }

                    TicToc t_solver;
                    ceres::Solver::Options options;
                    options.linear_solver_type = ceres::DENSE_QR;
                    options.max_num_iterations = 4;
                    options.minimizer_progress_to_stdout = false;
                    ceres::Solver::Summary summary;
                    ceres::Solve(options, &problem, &summary);
                    printf("solver time %f ms \n", t_solver.toc());
                }
                printf("optimization twice time %f \n", t_opt.toc());

                t_w_curr = t_w_curr + q_w_curr * t_last_curr;
                q_w_curr = q_w_curr * q_last_curr;
            }

            TicToc t_pub;

            // publish odometry 输出雷达里程计
            nav_msgs::Odometry laserOdometry;
            laserOdometry.header.frame_id = "/camera_init";
            laserOdometry.child_frame_id = "/laser_odom";
            laserOdometry.header.stamp = ros::Time().fromSec(timeSurfPointsLessFlat);
            laserOdometry.pose.pose.orientation.x = q_w_curr.x();//第一帧初始化时就是单位阵
            laserOdometry.pose.pose.orientation.y = q_w_curr.y();
            laserOdometry.pose.pose.orientation.z = q_w_curr.z();
            laserOdometry.pose.pose.orientation.w = q_w_curr.w();
            laserOdometry.pose.pose.position.x = t_w_curr.x();
            laserOdometry.pose.pose.position.y = t_w_curr.y();
            laserOdometry.pose.pose.position.z = t_w_curr.z();
            pubLaserOdometry.publish(laserOdometry);

            //输出整个轨迹，值就是odometry
            geometry_msgs::PoseStamped laserPose;
            laserPose.header = laserOdometry.header;
            laserPose.pose = laserOdometry.pose.pose;
            laserPath.header.stamp = laserOdometry.header.stamp;
            laserPath.poses.push_back(laserPose);
            laserPath.header.frame_id = "/camera_init";
            pubLaserPath.publish(laserPath);

            // transform corner features and plane features to the scan end point
            if (0)//总不执行? 对于kitti本身是去扭转的
            {
                int cornerPointsLessSharpNum = cornerPointsLessSharp->points.size();
                for (int i = 0; i < cornerPointsLessSharpNum; i++)
                {
                    TransformToEnd(&cornerPointsLessSharp->points[i], &cornerPointsLessSharp->points[i]);
                }

                int surfPointsLessFlatNum = surfPointsLessFlat->points.size();
                for (int i = 0; i < surfPointsLessFlatNum; i++)
                {
                    TransformToEnd(&surfPointsLessFlat->points[i], &surfPointsLessFlat->points[i]);
                }

                int laserCloudFullResNum = laserCloudFullRes->points.size();
                for (int i = 0; i < laserCloudFullResNum; i++)
                {
                    TransformToEnd(&laserCloudFullRes->points[i], &laserCloudFullRes->points[i]);
                }
            }
        
            //将cornerPointsLessSharp与laserCloudCornerLast交换，目的是保存cornerPointsLessSharp的值下轮使用
            pcl::PointCloud<PointType>::Ptr laserCloudTemp = cornerPointsLessSharp;
            cornerPointsLessSharp = laserCloudCornerLast;
            laserCloudCornerLast = laserCloudTemp;
            //将cornerPointsLessFlat与laserCloudSurfLast交换，目的是保存cornerPointsLessFlat的值下轮使用
            laserCloudTemp = surfPointsLessFlat;
            surfPointsLessFlat = laserCloudSurfLast;
            laserCloudSurfLast = laserCloudTemp;

            laserCloudCornerLastNum = laserCloudCornerLast->points.size();
            laserCloudSurfLastNum = laserCloudSurfLast->points.size();

            std::cout << "the size of corner last is " << laserCloudCornerLastNum << ", and the size of surf last is " << laserCloudSurfLastNum << '\n';

            //使用上一帧的特征点构建kd-tree
            kdtreeCornerLast->setInputCloud(laserCloudCornerLast);
            kdtreeSurfLast->setInputCloud(laserCloudSurfLast);

            //对点云undistort,转为一次扫描末尾的坐标系下（对kitti,有没有这一步都一样）
            int laserCloudFullResNum = laserCloudFullRes->points.size();//所有点的数量
            // //非地面点
            // pcl::PointCloud<PointType> laserCloudWOGround;
            // //地面点
            // pcl::PointCloud<PointType> laserCloudGround;
            
            for (int i = 0; i < laserCloudFullResNum; i++)
            {
                TransformToEnd(&laserCloudFullRes->points[i], &laserCloudFullRes->points[i]);
            
                // float height = (laserCloudFullRes->points[i]).z;//这样取对吗？
                // if (height > GroundThs)
                // {
                //     // GroundptsFlag[i] = 1;//高度高于阈值 即认为非地面点
                //     laserCloudWOGround.push_back(laserCloudFullRes->points[i]);//存入非地面点云中
                // }
                // else
                // {
                //     laserCloudGround.push_back(laserCloudFullRes->points[i]);
                // }
                
            }
            //从得到位姿到去除egomotion的时间
            printf("scan egomotion time %f ms \n", t_ego.toc());
            // printf("Ground point size : %d \n", laserCloudGround.points.size());

            //******************IMLS-SLAM的DYNAMIC OBJECT REMOVAL 
            TicToc t_remoground;
            //提取的地面点
            pcl::PointCloud<PointType>::Ptr laserCloudGround(new pcl::PointCloud<PointType>());
            //去除地面的点云
            pcl::PointCloud<PointType>::Ptr laserCloudWOGround(new pcl::PointCloud<PointType>());


            pcl::ModelCoefficients::Ptr coefficients (new pcl::ModelCoefficients);
            pcl::PointIndices::Ptr inliers (new pcl::PointIndices);
            // Create the segmentation object
            pcl::SACSegmentation<pcl::PointXYZI> segground;
            segground.setOptimizeCoefficients (true);//可选
            segground.setModelType (pcl::SACMODEL_PLANE);//设置分割模型类别:平面
            segground.setMethodType (pcl::SAC_RANSAC);//设置用哪个随机参数估计方法
            segground.setDistanceThreshold (0.15);//点到估计模型的距离最大值 可调节
            // segground.setMaxIterations (100);  //设置最大迭代次数

            segground.setInputCloud (laserCloudFullRes); //待分割的点云
            segground.segment (*inliers, *coefficients);
            if (inliers->indices.size () == 0)
            {
                PCL_ERROR ("Could not estimate a planar model for the given cloud frame.");
            }
            else
            {
                std::cout << "the size of Ground point is " << inliers->indices.size () << '\n';
            }

            // 提取地面
            pcl::ExtractIndices<pcl::PointXYZI> extract;//创建点云提取对象
            extract.setInputCloud (laserCloudFullRes);//设置输入点云
            extract.setIndices (inliers);//设置分割后的内点（地面点）为需要提取的点集
            extract.setNegative (false); //设置提取内点而非外点
            extract.filter (*laserCloudGround);//提取输出存储到laserCloudGround

            // 提取除地面外的点云
            extract.setNegative (true);
            extract.filter (*laserCloudWOGround);//提取输出存储到laserCloudWOGround
            //这里的全部点云就是上个文件scanregistration中采样后的点数,不再显示
            std::cout << ", and the size of other is " << laserCloudWOGround->points.size() << '\n';
            // printf("Remove ground points time %f ms \n", t_remoground.toc());

            //***开始 去除动态物体(small object removal)  基于聚类
            // 地面上的点云团　使用　欧式聚类的算法　kd树搜索　对点云聚类分割
            pcl::search::KdTree<pcl::PointXYZI>::Ptr stree (new pcl::search::KdTree<pcl::PointXYZI>);
            stree->setInputCloud (laserCloudWOGround);//　地面上其他的点云
            std::vector<pcl::PointIndices> cluster_indices;// 点云团索引
            pcl::EuclideanClusterExtraction<pcl::PointXYZI> ec;// 欧式聚类对象
            ec.setClusterTolerance (0.50); // 设置近邻搜索的搜索半径为0.5m paper的参数
            //ec.setMinClusterSize (100);  // 设置一个聚类需要的最少的点数目为100 不设限(default=1)，因为要保留出动态物体点云团之外的所有点
            //ec.setMaxClusterSize (25000); // 设置一个聚类需要的最大点数目为25000
            ec.setSearchMethod (stree); // 设置点云的搜索机制
            ec.setInputCloud (laserCloudWOGround);
            ec.extract (cluster_indices);           //从点云中提取聚类，并将点云索引保存在cluster_indices中
            
            //聚类后的所有点云团
            pcl::PointCloud<PointType>::Ptr cloud_cluster_all (new pcl::PointCloud<PointType>());
            //迭代访问点云索引cluster_indices,直到分割处所有聚类
            int j = 0;
            for (std::vector<pcl::PointIndices>::const_iterator it = cluster_indices.begin (); it != cluster_indices.end (); ++it)
            {
                pcl::PointCloud<PointType>::Ptr cloud_cluster (new pcl::PointCloud<PointType>());//每个group
                for (std::vector<int>::const_iterator pit = it->indices.begin (); pit != it->indices.end (); ++pit)
                {   
                    cloud_cluster->points.push_back (laserCloudWOGround->points[*pit]); //获取每一个点云团　的　点
                }
                cloud_cluster->width = cloud_cluster->points.size ();
                cloud_cluster->height = 1;
                cloud_cluster->is_dense = true;
                //std::cout << "the " << j << "th cloud cluster point number:" << cloud_cluster->points.size() << '\n';

                //计算该group的bounding box x,y,z
                PointType minPt, maxPt;
                pcl::getMinMax3D (*cloud_cluster, minPt, maxPt);
                float cloud_cluster_x = maxPt.x - minPt.x;
                float cloud_cluster_y = maxPt.y - minPt.y;
                float cloud_cluster_z = maxPt.z - minPt.z;

                //std::cout << "x range: "<< cloud_cluster_x << ", y range: " << cloud_cluster_y << ", z range: " << cloud_cluster_z << '\n';

                //若该group小于指定的box，不保留，跳过,判断下一个 points group 
                //关于如何准确的聚类到移动物体，判断条件可能需要调整  size界限越大，保留地越多
                if ( cloud_cluster_x <= Xbox && cloud_cluster_y <= Ybox && cloud_cluster_z <= Zbox && (cloud_cluster->points.size() >= 100) )
                {
                    //std::cout << "the " << j << "th cloud cluster are removed !" << '\n';
                    j++;
                    continue;
                }
                j++;
                //保存到总点云
                *cloud_cluster_all += *cloud_cluster;
                
            }

            pcl::PointCloud<PointType>::Ptr cloud_cluster_allWGround (new pcl::PointCloud<PointType>());
            *cloud_cluster_allWGround = *cloud_cluster_all;
            //添加回地面 得到了去除动态物体后的点云 cloud_4
            *cloud_cluster_allWGround += *laserCloudGround;
            //*************************************************
  


            //每隔skip帧才发给Mapping部分 这里还是1 10hz
            if (frameCount % skipFrameNum == 0)
            {
                frameCount = 0;
                //***
                //把三类点云发布到对应topic节点
                sensor_msgs::PointCloud2 laserCloudCornerLast2;
                pcl::toROSMsg(*laserCloudCornerLast, laserCloudCornerLast2);
                laserCloudCornerLast2.header.stamp = ros::Time().fromSec(timeSurfPointsLessFlat);
                laserCloudCornerLast2.header.frame_id = "/camera";
                pubLaserCloudCornerLast.publish(laserCloudCornerLast2);

                sensor_msgs::PointCloud2 laserCloudSurfLast2;
                pcl::toROSMsg(*laserCloudSurfLast, laserCloudSurfLast2);
                laserCloudSurfLast2.header.stamp = ros::Time().fromSec(timeSurfPointsLessFlat);
                laserCloudSurfLast2.header.frame_id = "/camera";
                pubLaserCloudSurfLast.publish(laserCloudSurfLast2);
                /***/
                sensor_msgs::PointCloud2 laserCloudFullRes3;
                pcl::toROSMsg(*laserCloudFullRes, laserCloudFullRes3);
                laserCloudFullRes3.header.stamp = ros::Time().fromSec(timeLaserCloudFullRes);
                laserCloudFullRes3.header.frame_id = "/camera";
                pubLaserCloudFullRes.publish(laserCloudFullRes3);

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
        }
        rate.sleep();
    }
    return 0;
}