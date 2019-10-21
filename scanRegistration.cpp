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
#include <vector>
#include <string>
#include "aloam_velodyne/common.h"
#include "aloam_velodyne/tic_toc.h" //关于时间长度
#include <nav_msgs/Odometry.h>
#include <opencv/cv.h>
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

using std::atan2;
using std::cos;
using std::sin;

//扫描周期，10hz数据，周期0.1s
const double scanPeriod = 0.1;

//初始化控制变量
const int systemDelay = 2; // 弃用前x帧，不弃用 new feature: only first 10 frames to find edge$planar feature 4662
int systemInitCount = 0;// new:计数当前处理的帧序号
bool systemInited = false;
//扫描线数？怎么是0？
int N_SCANS = 0;

//点云曲率，400000为一帧点云中最大数量
float cloudCurvature[400000];
//序号
int cloudSortInd[400000];
//点是否被筛选过标志 0-未选；1-已筛选
int cloudNeighborPicked[400000];
//点分类标号 根据曲率大小
int cloudLabel[400000];

//比较两点曲率
bool comp (int i,int j) { return (cloudCurvature[i]<cloudCurvature[j]); }

ros::Publisher pubLaserCloud;
ros::Publisher pubCornerPointsSharp;
ros::Publisher pubCornerPointsLessSharp;
ros::Publisher pubSurfPointsFlat;
ros::Publisher pubSurfPointsLessFlat;
ros::Publisher pubRemovePoints; //去除点
std::vector<ros::Publisher> pubEachScan;

bool PUB_EACH_LINE = false;//用于调试

double MINIMUM_RANGE = 0.1; //距离？
//类型模板声明 PoinT
template <typename PointT>
//去除较近点云
void removeClosedPointCloud(const pcl::PointCloud<PointT> &cloud_in,
                              pcl::PointCloud<PointT> &cloud_out, float thres)
{
    if (&cloud_in != &cloud_out)
    {
        cloud_out.header = cloud_in.header;
        cloud_out.points.resize(cloud_in.points.size());
    }

    size_t j = 0;

    for (size_t i = 0; i < cloud_in.points.size(); ++i)
    {//只保留thres之外的点
        if (cloud_in.points[i].x * cloud_in.points[i].x + cloud_in.points[i].y * cloud_in.points[i].y + cloud_in.points[i].z * cloud_in.points[i].z < thres * thres)
            continue;
        cloud_out.points[j] = cloud_in.points[i];
        j++;
    }
    if (j != cloud_in.points.size())
    {
        cloud_out.points.resize(j);
    }

    cloud_out.height = 1;
    cloud_out.width = static_cast<uint32_t>(j);
    cloud_out.is_dense = true;
}

//从topic/velodyne_point获得消息中的输入点云；提取各类特征点
void laserCloudHandler(const sensor_msgs::PointCloud2ConstPtr &laserCloudMsg)
{
    // if (!systemInited)
    // {//丢弃前systemDelay帧点云 （为了确保有imu数据，但该版本不使用imu，所以不丢弃）
    //     systemInitCount++;
    //     if (systemInitCount >= systemDelay) //1>=0
    //     {
    //         systemInited = true;
    //     }
    //     else
    //         return;
    // }

    TicToc t_whole;
    TicToc t_prepare;
    //记录每条scan有曲率的点的开始和结束索引
    std::vector<int> scanStartInd(N_SCANS, 0);//（64,0）
    std::vector<int> scanEndInd(N_SCANS, 0);

    //消息转为点云laserCloudIn
    pcl::PointCloud<pcl::PointXYZ> laserCloudIn;
    pcl::fromROSMsg(*laserCloudMsg, laserCloudIn);
    std::vector<int> indices;

    //去除输入点云的无效点；去除MINIMUM_RANGE之内的点云
    pcl::removeNaNFromPointCloud(laserCloudIn, laserCloudIn, indices);
    removeClosedPointCloud(laserCloudIn, laserCloudIn, MINIMUM_RANGE);


    int cloudSize = laserCloudIn.points.size();//点云数量
    //起始点、终止点的旋转角
    float startOri = -atan2(laserCloudIn.points[0].y, laserCloudIn.points[0].x);
    float endOri = -atan2(laserCloudIn.points[cloudSize - 1].y,
                          laserCloudIn.points[cloudSize - 1].x) +
                   2 * M_PI;

    //结束点与初始点的方位角差值控制在[pi,3pi] 为什么？
    if (endOri - startOri > 3 * M_PI)
    {
        endOri -= 2 * M_PI;
    }
    else if (endOri - startOri < M_PI)
    {
        endOri += 2 * M_PI;
    }
    //printf("end Ori %f\n", endOri);
    //lidar扫描线是否过半？ 半圆
    bool halfPassed = false;
    
    //把点云归为64线束*
    int count = cloudSize;
    PointType point;
    std::vector<pcl::PointCloud<PointType>> laserCloudScans(N_SCANS);//64线扫描
    for (int i = 0; i < cloudSize; i++)
    {   //kitti lidar坐标系
        point.x = laserCloudIn.points[i].x;
        point.y = laserCloudIn.points[i].y;
        point.z = laserCloudIn.points[i].z;

        //点的仰角
        float angle = atan(point.z / sqrt(point.x * point.x + point.y * point.y)) * 180 / M_PI;
        int scanID = 0; //点输入的扫描光束

        if (N_SCANS == 16)
        {
            scanID = int((angle + 15) / 2 + 0.5);//四舍五入
            if (scanID > (N_SCANS - 1) || scanID < 0)
            {
                count--;
                continue;
            }
        }
        else if (N_SCANS == 32)
        {
            scanID = int((angle + 92.0/3.0) * 3.0 / 4.0);
            if (scanID > (N_SCANS - 1) || scanID < 0)
            {
                count--;
                continue;
            }
        }
        else if (N_SCANS == 64)
        {   
            if (angle >= -8.83) //这个值是怎么来的， 24.33 8.83
                scanID = int((2 - angle) * 3.0 + 0.5);
            else
                scanID = N_SCANS / 2 + int((-8.83 - angle) * 2.0 + 0.5);

            // use [0 50]  > 50 remove outlies 过滤点 只挑选[0，50]线的点
            if (angle > 2 || angle < -24.33 || scanID > 50 || scanID < 0)
            {
                count--;
                continue;//下个点
            }
        }
        else
        {
            printf("wrong scan number\n");
            ROS_BREAK();
        }
        // printf("yang angle %f scanID %d \n", angle, scanID);//输出仰角和光束id

        float ori = -atan2(point.y, point.x);//该点方位角
        if (!halfPassed)//初始false
        {   //确保ori - startOri在[-pi/2,3pi/2] 为何是这个范围？
            if (ori < startOri - M_PI / 2)
            {
                ori += 2 * M_PI;
            }
            else if (ori > startOri + M_PI * 3 / 2)
            {
                ori -= 2 * M_PI;
            }

            if (ori - startOri > M_PI)
            {//超过半圆
                halfPassed = true;
            }
        }
        else
        {
            ori += 2 * M_PI;
            //确保ori - endOri在[-3pi/2,pi/2]
            if (ori < endOri - M_PI * 3 / 2)
            {
                ori += 2 * M_PI;
            }
            else if (ori > endOri + M_PI / 2)
            {
                ori -= 2 * M_PI;
            }
        }
        //[-0.5,1.5] 点旋转的角度与整个周期旋转角度的比率，即点云中点的相对扫描时间 为何有负值？
        float relTime = (ori - startOri) / (endOri - startOri);
        point.intensity = scanID + scanPeriod * relTime;//该点的强度=线号+点的时间(小数) 没有使用原始数据中的intensity(是什么)
        laserCloudScans[scanID].push_back(point);//把点分别存入对应的线号的容器保存
    }
    
    cloudSize = count;//过滤后的点数，有效范围内的所有点
    printf("points size after pre-process %d \n", cloudSize);

    pcl::PointCloud<PointType>::Ptr laserCloud(new pcl::PointCloud<PointType>());
    for (int i = 0; i < N_SCANS; i++)
    {//因为曲率的计算需要点前后各使用5个点，所有每个线号起始点跳过前5个，末尾点去掉后5个，存储了每线点有曲率的点的索引（有用吗）
        scanStartInd[i] = laserCloud->size() + 5;
        *laserCloud += laserCloudScans[i];//将所有点按照线号从小到大放入一个容器
        scanEndInd[i] = laserCloud->size() - 6;
    }

    systemInitCount++;
    //************ 对于imls-slam不需要提取这些特征点 但是为了undistort(kitti 不需要这个),需要一个位姿的估计
    if (systemInitCount > systemDelay) //?>10
    {
        if (!systemInited)
        {
            printf("scanRegis: Initialize Done after compute first %d frames edge&planar points ! \n", systemDelay);
            systemInited = true;
        }
        
        // return; //哈哈，这里怎么敢return！
    }
    else
    {//前10帧还需要计算4类特征点 并发布
        printf("prepare time before select edge&planar %f \n", t_prepare.toc());//提取特征前的准备时间
        //计算曲率
        for (int i = 5; i < cloudSize - 5; i++)//特征点只从[0,50]中挑选
        { //使用每个点的前后5个点计算曲率，因此前后5点均跳过
            float diffX = laserCloud->points[i - 5].x + laserCloud->points[i - 4].x + laserCloud->points[i - 3].x + laserCloud->points[i - 2].x + laserCloud->points[i - 1].x - 10 * laserCloud->points[i].x + laserCloud->points[i + 1].x + laserCloud->points[i + 2].x + laserCloud->points[i + 3].x + laserCloud->points[i + 4].x + laserCloud->points[i + 5].x;
            float diffY = laserCloud->points[i - 5].y + laserCloud->points[i - 4].y + laserCloud->points[i - 3].y + laserCloud->points[i - 2].y + laserCloud->points[i - 1].y - 10 * laserCloud->points[i].y + laserCloud->points[i + 1].y + laserCloud->points[i + 2].y + laserCloud->points[i + 3].y + laserCloud->points[i + 4].y + laserCloud->points[i + 5].y;
            float diffZ = laserCloud->points[i - 5].z + laserCloud->points[i - 4].z + laserCloud->points[i - 3].z + laserCloud->points[i - 2].z + laserCloud->points[i - 1].z - 10 * laserCloud->points[i].z + laserCloud->points[i + 1].z + laserCloud->points[i + 2].z + laserCloud->points[i + 3].z + laserCloud->points[i + 4].z + laserCloud->points[i + 5].z;
            //曲率计算 式1 （没有归一化项）
            cloudCurvature[i] = diffX * diffX + diffY * diffY + diffZ * diffZ;
            //曲率点的索引
            cloudSortInd[i] = i;
            //初始时，点全未被筛选过
            cloudNeighborPicked[i] = 0;
            //初始化为 less flat点
            cloudLabel[i] = 0;
        }
        //没有再去挑选点？
        TicToc t_pts;
        //根据曲率划分为4种类型的点
        pcl::PointCloud<PointType> cornerPointsSharp;
        pcl::PointCloud<PointType> cornerPointsLessSharp;
        pcl::PointCloud<PointType> surfPointsFlat;
        pcl::PointCloud<PointType> surfPointsLessFlat;

        float t_q_sort = 0;
        for (int i = 0; i < N_SCANS; i++)//遍历每个线，将每条线的点归为以上对应类别
        {
            if( scanEndInd[i] - scanStartInd[i] < 6)
                continue;//若该线的曲率起始点和终止点距离小于5个点，跳过该线？
            pcl::PointCloud<PointType>::Ptr surfPointsLessFlatScan(new pcl::PointCloud<PointType>);
            //将每个scan的曲率点分为6等份处理，确保周围都有点被选作特征点
            for (int j = 0; j < 6; j++)
            {   //6等份的起点和终点索引
                int sp = scanStartInd[i] + (scanEndInd[i] - scanStartInd[i]) * j / 6; 
                int ep = scanStartInd[i] + (scanEndInd[i] - scanStartInd[i]) * (j + 1) / 6 - 1;

                TicToc t_tmp;
                //曲率从小到大排序
                std::sort (cloudSortInd + sp, cloudSortInd + ep + 1, comp);
                t_q_sort += t_tmp.toc();

                //挑选每部分曲率很大和比较大的点
                int largestPickedNum = 0;
                for (int k = ep; k >= sp; k--)
                {
                    int ind = cloudSortInd[k];//曲率最大点的索引 

                    if (cloudNeighborPicked[ind] == 0 &&
                        cloudCurvature[ind] > 0.1)
                    {//若曲率的确较大且未被筛选过

                        largestPickedNum++;
                        if (largestPickedNum <= 2)
                        {//挑选曲率最大的前2个点放入sharp点集合，也是lesssharp点                        
                            cloudLabel[ind] = 2;
                            cornerPointsSharp.push_back(laserCloud->points[ind]);
                            cornerPointsLessSharp.push_back(laserCloud->points[ind]);
                        }
                        else if (largestPickedNum <= 20)
                        {//挑选曲率最大的前20个点放入less sharp点集合                        
                            cloudLabel[ind] = 1; 
                            cornerPointsLessSharp.push_back(laserCloud->points[ind]);
                        }
                        else
                        {
                            break;
                        }

                        cloudNeighborPicked[ind] = 1;//筛选标志置位 

                        //将曲率比较大的点的前后各5个连续距离比较近的点 筛选出去，防止特征点聚集
                        for (int l = 1; l <= 5; l++)
                        {
                            float diffX = laserCloud->points[ind + l].x - laserCloud->points[ind + l - 1].x;
                            float diffY = laserCloud->points[ind + l].y - laserCloud->points[ind + l - 1].y;
                            float diffZ = laserCloud->points[ind + l].z - laserCloud->points[ind + l - 1].z;
                            if (diffX * diffX + diffY * diffY + diffZ * diffZ > 0.05)
                            {
                                break;
                            }

                            cloudNeighborPicked[ind + l] = 1;
                        }
                        for (int l = -1; l >= -5; l--)
                        {
                            float diffX = laserCloud->points[ind + l].x - laserCloud->points[ind + l + 1].x;
                            float diffY = laserCloud->points[ind + l].y - laserCloud->points[ind + l + 1].y;
                            float diffZ = laserCloud->points[ind + l].z - laserCloud->points[ind + l + 1].z;
                            if (diffX * diffX + diffY * diffY + diffZ * diffZ > 0.05)
                            {
                                break;
                            }

                            cloudNeighborPicked[ind + l] = 1;
                        }
                    }
                }

                int smallestPickedNum = 0;
                for (int k = sp; k <= ep; k++)
                {
                    int ind = cloudSortInd[k];

                    if (cloudNeighborPicked[ind] == 0 &&
                        cloudCurvature[ind] < 0.1)//若曲率的确较小且未被筛选过
                    {

                        cloudLabel[ind] = -1; //只选曲率最小的4个点作为 flat
                        surfPointsFlat.push_back(laserCloud->points[ind]);

                        smallestPickedNum++;
                        if (smallestPickedNum >= 4)
                        { 
                            break;
                        }

                        cloudNeighborPicked[ind] = 1;//筛选标志置位
                        //将曲率比较小的点的前后各5个连续距离比较近的点 筛选出去，防止特征点聚集
                        for (int l = 1; l <= 5; l++)
                        { 
                            float diffX = laserCloud->points[ind + l].x - laserCloud->points[ind + l - 1].x;
                            float diffY = laserCloud->points[ind + l].y - laserCloud->points[ind + l - 1].y;
                            float diffZ = laserCloud->points[ind + l].z - laserCloud->points[ind + l - 1].z;
                            if (diffX * diffX + diffY * diffY + diffZ * diffZ > 0.05)
                            {
                                break;
                            }

                            cloudNeighborPicked[ind + l] = 1;
                        }
                        for (int l = -1; l >= -5; l--)
                        {
                            float diffX = laserCloud->points[ind + l].x - laserCloud->points[ind + l + 1].x;
                            float diffY = laserCloud->points[ind + l].y - laserCloud->points[ind + l + 1].y;
                            float diffZ = laserCloud->points[ind + l].z - laserCloud->points[ind + l + 1].z;
                            if (diffX * diffX + diffY * diffY + diffZ * diffZ > 0.05)
                            {
                                break;
                            }

                            cloudNeighborPicked[ind + l] = 1;
                        }
                    }
                }
                //把剩余的点（包括前面被排除的点）都归为less flat
                for (int k = sp; k <= ep; k++)
                {
                    if (cloudLabel[k] <= 0)
                    {
                        surfPointsLessFlatScan->push_back(laserCloud->points[k]);
                    }
                }
            }
            //对less flat体素栅格滤波
            pcl::PointCloud<PointType> surfPointsLessFlatScanDS;
            pcl::VoxelGrid<PointType> downSizeFilter;
            downSizeFilter.setInputCloud(surfPointsLessFlatScan);
            downSizeFilter.setLeafSize(0.2, 0.2, 0.2);
            downSizeFilter.filter(surfPointsLessFlatScanDS);
            //less flat点汇总
            surfPointsLessFlat += surfPointsLessFlatScanDS;
        }

        printf("sort q time %f \n", t_q_sort);//排序时间
        printf("seperate points feature time %f \n", t_pts.toc());//提取特征点用时

        sensor_msgs::PointCloud2 cornerPointsSharpMsg;
        pcl::toROSMsg(cornerPointsSharp, cornerPointsSharpMsg);
        cornerPointsSharpMsg.header.stamp = laserCloudMsg->header.stamp;
        cornerPointsSharpMsg.header.frame_id = "/camera_init";
        pubCornerPointsSharp.publish(cornerPointsSharpMsg);

        sensor_msgs::PointCloud2 cornerPointsLessSharpMsg;
        pcl::toROSMsg(cornerPointsLessSharp, cornerPointsLessSharpMsg);
        cornerPointsLessSharpMsg.header.stamp = laserCloudMsg->header.stamp;
        cornerPointsLessSharpMsg.header.frame_id = "/camera_init";
        pubCornerPointsLessSharp.publish(cornerPointsLessSharpMsg);

        sensor_msgs::PointCloud2 surfPointsFlat2;
        pcl::toROSMsg(surfPointsFlat, surfPointsFlat2);
        surfPointsFlat2.header.stamp = laserCloudMsg->header.stamp;
        surfPointsFlat2.header.frame_id = "/camera_init";
        pubSurfPointsFlat.publish(surfPointsFlat2);

        sensor_msgs::PointCloud2 surfPointsLessFlat2;
        pcl::toROSMsg(surfPointsLessFlat, surfPointsLessFlat2);
        surfPointsLessFlat2.header.stamp = laserCloudMsg->header.stamp;
        surfPointsLessFlat2.header.frame_id = "/camera_init";
        pubSurfPointsLessFlat.publish(surfPointsLessFlat2);

    }
    

    /********************/

    sensor_msgs::PointCloud2 laserCloudOutMsg;
    pcl::toROSMsg(*laserCloud, laserCloudOutMsg);//过滤后的点，按线储存的 将传入下个节点
    laserCloudOutMsg.header.stamp = laserCloudMsg->header.stamp;
    laserCloudOutMsg.header.frame_id = "/camera_init";
    pubLaserCloud.publish(laserCloudOutMsg);
    //*******************
    
    /************************************************************************/
    // pub each scam
    if(PUB_EACH_LINE)
    {
        for(int i = 0; i< N_SCANS; i++)
        {
            sensor_msgs::PointCloud2 scanMsg;
            pcl::toROSMsg(laserCloudScans[i], scanMsg);
            scanMsg.header.stamp = laserCloudMsg->header.stamp;
            scanMsg.header.frame_id = "/camera_init";
            pubEachScan[i].publish(scanMsg);
        }
    }

    printf("whole scan registration time %f ms ***********************\n \n", t_whole.toc()); //整个scan的注册过程用时
    if(t_whole.toc() > 100)
        ROS_WARN("scan registration process over 100ms");
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "scanRegistration");
    ros::NodeHandle nh;

    nh.param<int>("scan_line", N_SCANS, 16);// "scan_line"=64的值给N_SCANS 来自于aloam_velodyne_HDL_64.launch （kitti）

    nh.param<double>("minimum_range", MINIMUM_RANGE, 0.1);//5 (kitti)

    printf("scan line number %d \n", N_SCANS);//扫描线数
    printf("minimum range %f \n", MINIMUM_RANGE); //确认参数

    if(N_SCANS != 16 && N_SCANS != 32 && N_SCANS != 64)
    {
        printf("only support velodyne with 16, 32 or 64 scan line!");
        return 0;
    }
    //订阅“/velodyne_points”节点,调用函数 laserCloudHandler 接受输入的激光点云消息
    ros::Subscriber subLaserCloud = nh.subscribe<sensor_msgs::PointCloud2>("/velodyne_points", 100, laserCloudHandler);
    //在对应topic发布 点云类型的消息
    pubLaserCloud = nh.advertise<sensor_msgs::PointCloud2>("/velodyne_cloud_2", 100);//处理后的总点云，传给下个节点

    //***** to do 以下四个节点 不一定发布 只在前10帧才有发布
    pubCornerPointsSharp = nh.advertise<sensor_msgs::PointCloud2>("/laser_cloud_sharp", 100);

    pubCornerPointsLessSharp = nh.advertise<sensor_msgs::PointCloud2>("/laser_cloud_less_sharp", 100);

    pubSurfPointsFlat = nh.advertise<sensor_msgs::PointCloud2>("/laser_cloud_flat", 100);

    pubSurfPointsLessFlat = nh.advertise<sensor_msgs::PointCloud2>("/laser_cloud_less_flat", 100);
    //******

    pubRemovePoints = nh.advertise<sensor_msgs::PointCloud2>("/laser_remove_points", 100);//没有发布啊 没有用到

    if(PUB_EACH_LINE)//默认false
    {//通过观察rviz，scan00表示较远处（俯仰角大的），scan50俯仰角小的，离车体最近的
        for(int i = 0; i < N_SCANS; i++) //每条线
        {
            ros::Publisher tmp = nh.advertise<sensor_msgs::PointCloud2>("/laser_scanid_" + std::to_string(i), 100);
            pubEachScan.push_back(tmp);
        }
    }
    ros::spin();

    return 0;
}
