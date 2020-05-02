
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
int mapCount = 0;//实际地图中的帧数
//时间戳
double timecloudprocessed = 0;
double timecloudGround = 0;
double timeLaserCloudCornerLast = 0;
double timeLaserCloudSurfLast = 0;
double timeLaserCloudFullRes = 0;
double timeLaserOdometry = 0;
float overlap_ratio;

//keep n =100(paper) scan in the map
#define mapsize 100 //1 3 5 10 20 40 50 60 70 100  
//从每个scan list中采样的点数s=100
#define numselect 100 // 因为kitti一帧数据太多了 665 650 800（0.7%）222 807（0.7%） 657 230 200
#define SampNeiThr 0.20 //计算特征点和地图中最近点法线 的近邻范围 原来是0.5  考虑自适应的选择？
#define SampNeiNum 15 //尝试估计法线时用近邻个数选邻域
//采样中的 outlier 判别阈值
#define OutlierThr 0.20  //0.2；再调的参数！ 0.2 0.18 0.18
#define RadThr 0.20 //计算I_x时那个box邻域半径 0.2 0.18 0.20
// #define numkneibor 5 //计算map 点法线 时近邻个数 5，8，9
//定义IMLS surface的参数 h
#define h_imls 0.06
//lossfunction 的阈值参数
#define lossarg 0.10 //huberloss 原始是0.1 0.2 
//ICP优化次数
#define numICP 21 //论文是20次 为了看20次ICP后可视化的结果 21>16
#define maxnumiter1ICP 4 //一次ICP中最大迭代次数（for ceres）
//保存不同k近邻下表示选择邻域大小的熵值
std::vector<float> Ef_k(9); //分为9分
//近邻大小的取值范围
std::vector<int> k_candi {4, 6, 8, 12, 20, 32, 47, 55, 64};
//表示float的无穷大
float infinity = (float)1.0 / 0.0;
//统一法线的方向 视点坐标的设置  会影响法线可视化
Eigen::Vector3d Viewpoint(0, 0, 0);

//距离滤波 最近 最远 阈值
#define distance_near_thresh 5 // loam 已经剔除了5m之内的点
#define distance_far_thresh 120 //待设置 kitti<81
//设置下采样的相关参数
#define downsample_method "VOXELGRID"  //APPROX_VOXELGRID VOXELGRID NONE
#define downsample_resolution 0.7 //0.1 0.2 0.6 0.5 0.7 0.8
//设置外点去除的相关参数
#define outlier_removal_method "RADIUS" //STATISTICAL RADIUS NONE
#define radius_radius 1.5
#define radius_min_neighbors 3
#define statistical_mean_k 20 //30
#define statistical_stddev 1.5 //1.2

//small group removal 界限
#define Xbox 14
#define Ybox 14
#define Zbox 4
#define minclusterRMThs 20 //box之内的点多于该阈值 才被去除
#define maxclusterRMThs 150
//0.1 0.2 <0.3<0.4<0.5<0.6<little0.7<0.8<0.9
#define matchrmperc 0.40 //去除对应点距离较差的 % 的点
//被选择用来匹配的对应点的帧范围阈值
#define nearlastThs 1  //只有对应点在该范围内 才被用来优化 1 3>5>10 >15 >20 >30
//和最大帧距相乘 取较近区间:0.05>0.1>0.2>0.3>0.4>0.5>0.6  取较远区间0.1 0.2 0.3 0.4 0.5 0.6 该方式已测完
#define fdistrmperc 0.60  //排序分位数 去除帧距较远的x%点 0.1<0.2<0.3<0.4<0.5<0.6>0.7
bool IRmSome = false;//是否考虑按residual大小去除
#define I_rmperc 0.20 //排序分位数 去除到隐式平面的距离较大的x%
//存储地图的间隔1>2>3>5>10
#define kframespace 1 
#define n_angleths 60.0 //法线方向夹角阈值 60 50 40 30 20 15 10
//定义最小overlap ratio 小于它才视为关键帧 加入地图 参考文献是0.75; 0.77  0.85
#define min_overlap 0.80

//当前帧采样之后用于maping的点  9*100=900
pcl::PointCloud<pointxyzinormal>::Ptr CloudSampled(new pcl::PointCloud<pointxyzinormal>());

//input & output: points in one frame. local --> global 一帧的所有点
pcl::PointCloud<PointType>::Ptr laserCloudFullRes(new pcl::PointCloud<PointType>());
pcl::PointCloud<PointType>::Ptr Cloudprocess_raw(new pcl::PointCloud<PointType>());
pcl::PointCloud<PointType>::Ptr CloudProcessed(new pcl::PointCloud<PointType>());
//当前扫描周围的部分地图
pcl::PointCloud<PointType>::Ptr laserCloudSurround(new pcl::PointCloud<PointType>());

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
//用于存储历史某个位姿 用于计算特征点对 匹配距离
// Eigen::Quaterniond q_wy(1, 0, 0, 0);
// Eigen::Vector3d t_wy(0, 0, 0);

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
//分离出的地面点，去除运动物体前的点云, 运动物体点云
ros::Publisher pubCloudGround, pubCloudbfDOR, pubDynamicObj;
//发布每类采样点 用于分析特征点的采样效果
std::vector<ros::Publisher> pubEachFeatList;
bool PUB_EACH_List = true;//用于调试 false true
nav_msgs::Path laserAfterMappedPath;

ofstream outcloudscan, outfeatpair, sortfeat, outangle, outoverlap;


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
//保存IMLS给定的结果轨迹
vector< Eigen::Isometry3d > pose_IMLS;
// kitti gt 对应的lidar坐标系下的轨迹
vector< Eigen::Isometry3d > pose_gt;
//保存历史位姿
vector< Eigen::Isometry3d > pose_h;

//读取gt的轨迹
bool loadPoses_gt(string file_name) 
{

  FILE *fp = fopen(file_name.c_str(),"r");
  if (!fp)
    return false;
  while (!feof(fp)) 
  {
    Eigen::Isometry3d P = Eigen::Isometry3d::Identity();
    
    if (fscanf(fp, "%lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf",
                   &P.matrix()(0,0), &P.matrix()(0,1), &P.matrix()(0,2), &P.matrix()(0,3),
                   &P.matrix()(1,0), &P.matrix()(1,1), &P.matrix()(1,2), &P.matrix()(1,3),
                   &P.matrix()(2,0), &P.matrix()(2,1), &P.matrix()(2,2), &P.matrix()(2,3) )==12)  
    {
		pose_gt.push_back(P);
    }
  }
  fclose(fp);
  return true;
}

//读取imls的轨迹
bool loadPoses(string file_name) 
{

  FILE *fp = fopen(file_name.c_str(),"r");
  if (!fp)
    return false;
  while (!feof(fp)) 
  {
    Eigen::Isometry3d P = Eigen::Isometry3d::Identity();
    
    if (fscanf(fp, "%lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf",
                   &P.matrix()(0,0), &P.matrix()(0,1), &P.matrix()(0,2), &P.matrix()(0,3),
                   &P.matrix()(1,0), &P.matrix()(1,1), &P.matrix()(1,2), &P.matrix()(1,3),
                   &P.matrix()(2,0), &P.matrix()(2,1), &P.matrix()(2,2), &P.matrix()(2,3) )==12)  
    {
		pose_IMLS.push_back(P);
    }
  }
  fclose(fp);
  return true;
}

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
	//使用变换矩阵来变换 想排除累积地图过程对结果的影响  会好一些..
	Eigen::Isometry3d T_w_curr = Eigen::Isometry3d::Identity();
	Eigen::AngleAxisd rotation_w_curr(q_w_curr);
	T_w_curr.rotate(rotation_w_curr);
	T_w_curr.pretranslate(t_w_curr);

	// Eigen::Vector3d point_w = q_w_curr * point_curr + t_w_curr;
	Eigen::Vector3d point_w = T_w_curr * point_curr;
	po->x = point_w.x();
	po->y = point_w.y();
	po->z = point_w.z();
	po->intensity = pi->intensity;
	//po->intensity = 1.0;
	po->normal[0] = pi->normal[0];//法线信息保留不变
	po->normal[1] = pi->normal[1];
	po->normal[2] = pi->normal[2];
}
//使用gt/imls去得到地图
void pointAssociateToMap_gt(PointType const *const pi, PointType *const po, int frameid)
{
	Eigen::Vector3d point_curr(pi->x, pi->y, pi->z);
	// Eigen::Vector3d point_w = q_w_curr * point_curr + t_w_curr;
	Eigen::Vector3d point_w = pose_IMLS[frameid] * point_curr;// or pose_gt  pose_IMLS
	po->x = point_w.x();
	po->y = point_w.y();
	po->z = point_w.z();
	po->intensity = frameid;//记录原来属于哪一帧
	// po->normal[0] = pi->normal[0];//法线信息保留不变
	// po->normal[1] = pi->normal[1];
	// po->normal[2] = pi->normal[2];
}

//转移到局部坐标系 是pointAssociateToMap（）的逆过程
void pointAssociateTobeMapped(pointxyzinormal const *const pi, pointxyzinormal *const po)
{
	Eigen::Vector3d point_w(pi->x, pi->y, pi->z);
	Eigen::Vector3d point_curr = q_w_curr.inverse() * (point_w - t_w_curr);
	po->x = point_curr.x();
	po->y = point_curr.y();
	po->z = point_curr.z();
	po->intensity = pi->intensity;
	po->normal[0] = pi->normal[0];//法线信息保留不变
	po->normal[1] = pi->normal[1];
	po->normal[2] = pi->normal[2];
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
    segground.setDistanceThreshold (0.5);//点到估计模型的距离最大值 可调节 0.15 0.5
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
void dynamic_object_removal(const pcl::PointCloud<PointType>::Ptr& src_cloud, pcl::PointCloud<PointType>::Ptr& cloud_cluster_all, pcl::PointCloud<PointType>::Ptr& rm_cloud)
{
    //聚类后的所有点云团 待输出
    // pcl::PointCloud<PointType>::Ptr cloud_cluster_all(new pcl::PointCloud<PointType>());
    pcl::search::KdTree<PointType>::Ptr stree (new pcl::search::KdTree<PointType>);
    stree->setInputCloud (src_cloud);//　不含地面的点云
    std::vector<pcl::PointIndices> cluster_indices;// 点云团索引
    pcl::EuclideanClusterExtraction<PointType> ec;// 欧式聚类对象
    ec.setClusterTolerance (0.90); // 设置近邻搜索的搜索半径为0.5m paper的参数 0.9
    ec.setMinClusterSize (10);  // 设置一个聚类需要的最少的点数目为100 不设限(default=1)，因为要保留出动态物体点云团之外的所有点 6
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
			*rm_cloud += *cloud_cluster;
            j++;
            continue;
        }
        j++;
        //保存到总点云
        *cloud_cluster_all += *cloud_cluster;
        
    }

    // return cloud_cluster_all;
}

//计算法线并保存为xyzinormal格式 输入是xyz格式
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

//给registered帧重新计算法线并保存为xyzinormal格式 输入是xyzinormal格式
pcl::PointCloud<pointxyzinormal>::Ptr recompute_normal( const pcl::PointCloud<pointxyzinormal>::Ptr& src_cloud )
{
    pcl::PointCloud<pointxyzinormal>::Ptr dst_cloud(new pcl::PointCloud<pointxyzinormal>);
    //当前点云建立kdtree
    pcl::KdTreeFLANN<pointxyzinormal>::Ptr kdtreeFromsrc(new pcl::KdTreeFLANN<pointxyzinormal>());
    kdtreeFromsrc->setInputCloud(src_cloud);
    std::vector<int> pointShInd;
    std::vector<float> pointShSqDis;
    int nump = src_cloud->points.size();

    for (int i=0; i<nump; i++)
    {
        pointxyzinormal pointtmp = src_cloud->points[i];
        
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
		pointout.intensity = pointtmp.intensity;//它本身inyensity是有值的 这里是帧序
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

			// CloudProcessed->clear();
			// if(int(CloudProcessed->points.size()) != 0)
			// {
			// 	std::cout<<"Failed to clear CloudProcessed !\n"<<endl;
			// }
			// CloudSampled.reset(new pcl::PointCloud<pointxyzinormal>());
			// if(int(CloudSampled->points.size()) != 0)
			// {
			// 	std::cout<<"Failed to clear CloudSampled !\n"<<endl;
			// }
			laserCloudFullRes->clear(); //在debug阶段 它和cloudprocessed其实一样
			// if(int(laserCloudFullRes->points.size()) != 0)
			// {
			// 	std::cout<<"Failed to clear laserCloudFullRes !\n"<<endl;
			// }
			pcl::fromROSMsg(*fullResBuf.front(), *laserCloudFullRes);
			fullResBuf.pop();

			// Cloudprocess_raw->clear();
			// pcl::fromROSMsg(*processedBuf.front(), *Cloudprocess_raw);
			// processedBuf.pop();

			//为了计算特征点匹配度 每一帧都用gt/imls结果
			// Eigen::Isometry3d Tgt = pose_gt[frameCount];
			// Eigen::Quaterniond qgt( Tgt.matrix().topLeftCorner<3, 3>() );
			// Eigen::Vector3d tgt = Tgt.matrix().topRightCorner<3, 1>(); 
			// q_wodom_curr = qgt;
			// t_wodom_curr = tgt;
			
			// 位姿的初始估计值 需要知道前两帧的优化后的最终位姿
			if( frameCount < 3 )
			{//第0帧，第1帧 位姿初始值都为I
				//第0帧 直接输出单位阵 q_wodom_curr初始值就是I
				if(true)//frameCount==1 || frameCount==2
				{
					Eigen::Isometry3d Tgt = pose_IMLS[frameCount]; //pose_gt  pose_IMLS
					Eigen::Quaterniond qgt( Tgt.matrix().topLeftCorner<3, 3>() );
					Eigen::Vector3d tgt = Tgt.matrix().topRightCorner<3, 1>(); //用真值与否不影响第一帧的匹配结果

					q_wodom_curr = qgt;
					t_wodom_curr = tgt;
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

				//debug: 使用真实位姿 作为初始值 用于单独查看算法匹配的效果
				// Eigen::Isometry3d Tgt_w_k_1 = pose_gt[frameCount-1];
				// Eigen::Isometry3d Tgt_w_k_2 = pose_gt[frameCount-2];
				// Eigen::Quaterniond qgt_w_k_1( Tgt_w_k_1.matrix().topLeftCorner<3, 3>() );
				// Eigen::Quaterniond qgt_w_k_2( Tgt_w_k_2.matrix().topLeftCorner<3, 3>() );
				// Eigen::Vector3d tgt_w_k_1(Tgt_w_k_1.matrix()(0,3), Tgt_w_k_1.matrix()(1,3), Tgt_w_k_1.matrix()(2,3));
				// Eigen::Vector3d tgt_w_k_2(Tgt_w_k_2.matrix()(0,3), Tgt_w_k_2.matrix()(1,3), Tgt_w_k_2.matrix()(2,3));

				// q_w_k_1 = qgt_w_k_1;
				// q_w_k_2 = qgt_w_k_2;
				// t_w_k_1 = tgt_w_k_1;
				// t_w_k_2 = tgt_w_k_2;
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
				// Eigen::Isometry3d T_wodom_curr = Tgt_w_k_1 * Tgt_w_k_2.inverse() * Tgt_w_k_1;

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
			
			std::cerr<<"the init 'w_curr' pose of frame"<<frameCount<<": q= "<<q_w_curr.coeffs().transpose()<<" t= "<<t_w_curr.transpose()<<"\n"<<endl;

			//----------------预处理---------------------//
			TicToc t_prefilter;
			int numprocess_raw = laserCloudFullRes->points.size();
			//1 距离滤波 去掉<5m的点
			pcl::PointCloud<PointType>::Ptr dist_filter(new pcl::PointCloud<PointType>());
			std::cout<<"distance filtering Scan "<<frameCount<<" ..."<<endl;
			dist_filter = distance_filter(laserCloudFullRes);//distance_filter(laserCloudFullRes)
			int numdistf = dist_filter->points.size();
    		std::cout<<"before prefilter: "<<numprocess_raw<<" ,after dist_filter: "<<numdistf<<endl;

			//2.下采样  0.6 or 0.7 ds
			pcl::PointCloud<PointType>::Ptr DS_filter(new pcl::PointCloud<PointType>());
			std::cout<<"Downsampling Scan "<<frameCount<<" ..."<<endl;
			DS_filter = downsample(dist_filter);
			int numds = DS_filter->points.size();
    		std::cout<<"after downsampled: "<<numds<<endl;

			//3.去除离群点
			pcl::PointCloud<PointType>::Ptr Outlierm(new pcl::PointCloud<PointType>());
			std::cout<<"Removing Outlier from Scan "<<frameCount<<" ..."<<endl;
			Outlierm = outlier_removal(DS_filter); //r 1.5 3 
			int numoutlierrm = Outlierm->points.size();
			std::cout<<"after Outliers removed: "<<numoutlierrm<<endl;

			printf("Scan point cloud prefiltering time %f ms\n", t_prefilter.toc());
			//-----------------动态物体去除-------------------------//  暂时关闭该模块
			
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
			pcl::PointCloud<PointType>::Ptr DynamicObj(new pcl::PointCloud<PointType>());
			std::cout<<"Remove Dynamic Objects from Scan_WOGround "<<frameCount<<" ..."<<endl;
			// rmDynamic = dynamic_object_removal(WOGround);
			dynamic_object_removal(WOGround, rmDynamic, DynamicObj);
			int numRMdynamic = rmDynamic->points.size();
			std::cout<<"after Removed Dynamic Objects: "<<numRMdynamic<<endl;

			printf("Dynamic Objects Removal time %f ms\n", t_dynamicobjectremoval.toc());
			//增加回地面点
			*rmDynamic += *Ground_cloud;
			*CloudProcessed = *rmDynamic;
			
			// *CloudProcessed = *Outlierm;
			int numProcessed = CloudProcessed->points.size();//经历预滤波和运动物体取出后的点数
			std::cout<<"Before All-Preprocess Scan "<<frameCount<<": "<<numprocess_raw<<" ,after: "<<numProcessed<<endl;
			
			TicToc t_getfeatv;
			//先计算当前帧点云的法线 数据格式变为pointxyzinormal
			pcl::PointCloud<pointxyzinormal>::Ptr ScanWNO(new pcl::PointCloud<pointxyzinormal>());
			
			std::cout<<"Computing normal for Scan-Processed "<<frameCount<<" ..."<<endl;
			//r0.2 k15
   	 		ScanWNO = compute_normal(CloudProcessed); //使用的是一些列预处理后的scan current  原坐标系下的法线
			std::cout<<"Computing 9 feature for Scan-Processed "<<frameCount<<" ..."<<endl;
			//先对特征数组 和排序表数组 清零
			memset(samplefeature1,0,sizeof(float) * 150000);
			memset(samplefeature2,0,sizeof(float) * 150000);
			memset(samplefeature3,0,sizeof(float) * 150000);
			memset(samplefeature4,0,sizeof(float) * 150000);
			memset(samplefeature5,0,sizeof(float) * 150000);
			memset(samplefeature6,0,sizeof(float) * 150000);
			memset(samplefeature7,0,sizeof(float) * 150000);
			memset(samplefeature8,0,sizeof(float) * 150000);
			memset(samplefeature9,0,sizeof(float) * 150000);
			memset(cloudSortInd1,0,sizeof(int) * 150000);
			memset(cloudSortInd2,0,sizeof(int) * 150000);
			memset(cloudSortInd3,0,sizeof(int) * 150000);
			memset(cloudSortInd4,0,sizeof(int) * 150000);
			memset(cloudSortInd5,0,sizeof(int) * 150000);
			memset(cloudSortInd6,0,sizeof(int) * 150000);
			memset(cloudSortInd7,0,sizeof(int) * 150000);
			memset(cloudSortInd8,0,sizeof(int) * 150000);
			memset(cloudSortInd9,0,sizeof(int) * 150000);
			//k opt [4,64] 从9个值中挑选
			compute_feature(ScanWNO);//点数还是numProcessed!
			printf("point features compute time %f ms\n", t_getfeatv.toc());

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

			//地图中的特征点数目满足要求  若是第0帧 不执行这部分 包含匹配以及优化
			std::vector<pcl::PointCloud<pointxyzinormal>> CloudSampledFeat(9);//9类特征点云数组
			pcl::PointCloud<pointxyzinormal>::Ptr ScanWN(new pcl::PointCloud<pointxyzinormal>());
			std::vector<int> fdistThs(9);//记录9个列表的帧距阈值
			double Idist_thr;//I的阈值
			int fdist_thr;//帧距的阈值
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
				int numpicked;
				int numicp = numICP;
				// if(frameCount <= 2)
				// {
				// 	numicp = numICP; //第1帧 第2帧 就拿真值试试 不在优化
				// }
				// else
				// {
				// 	numicp = numICP;
				// }
				
				for (int iterCount = 0; iterCount < numicp; iterCount++)  //debug
				{
					
					//每一次都对当前帧进行变换 重新计算法线 有必要吗
					pcl::PointCloud<PointType>::Ptr Cloudtmp(new pcl::PointCloud<PointType>());
					//对当前帧进行坐标变换 使用当前的结果位姿
					for (int i = 0; i < numProcessed; i++)
					{	
						pointAssociateToMap(&CloudProcessed->points[i], &pointSel);
						Cloudtmp->push_back(pointSel);
					}
					// TicToc t_getfeatv;
					ScanWN->clear();
					std::cout<<"Computing normal for Cloudtmp "<<frameCount<<" ..."<<endl;
					//r0.2 k15
					ScanWN = compute_normal(Cloudtmp); //使用的是新坐标系下的法线
					// std::cout<<"Computing 9 feature for Cloudtmp "<<frameCount<<" ..."<<endl;
					//特征值只在原坐标系计算一次 
					// compute_feature(ScanWN);//点数还是numProcessed!
					// printf("point features REcompute time %f ms\n", t_getfeatv.toc());

					//对9个表进行从大到小排序
					// std::sort (cloudSortInd1, cloudSortInd1 + numProcessed, comp1);
					// std::sort (cloudSortInd2, cloudSortInd2 + numProcessed, comp2);
					// std::sort (cloudSortInd3, cloudSortInd3 + numProcessed, comp3);
					// std::sort (cloudSortInd4, cloudSortInd4 + numProcessed, comp4);
					// std::sort (cloudSortInd5, cloudSortInd5 + numProcessed, comp5);
					// std::sort (cloudSortInd6, cloudSortInd6 + numProcessed, comp6);
					// std::sort (cloudSortInd7, cloudSortInd7 + numProcessed, comp7);
					// std::sort (cloudSortInd8, cloudSortInd8 + numProcessed, comp8);
					// std::sort (cloudSortInd9, cloudSortInd9 + numProcessed, comp9);
		
					if ( iterCount == 0 || iterCount == numicp-1 )//只保存numICP-1次匹配后的特征点以及对应点
					{
						// cout<<"saving point pair-> featpair-"<<iterCount<<"-s"<<numselect<<"-f"<<frameCount<<".txt ..."<<endl;
						// outfeatpair.open("/home/dlr/imlslam/txttemp/pairdist-" + //featpair-20-s222-f .txt
						// 					std::to_string(iterCount) + "-s" + std::to_string(numselect) + "-f" + 
						// 					std::to_string(frameCount) + ".txt");
						
						if (iterCount == numicp-1)//只在numICP-1次ICP后记录此时的scan overlap ratio
						{
							outoverlap.open("/home/dlr/imlslam/txttemp/overlap/op-m" + //op-20-s100.txt一次实验写入一个文件
											std::to_string(mapsize) + "-s" + std::to_string(numselect) + ".txt", ios::app);
							outangle.open("/home/dlr/imlslam/txttemp/nvangle-" + //nvangle-20-s100-f .txt
											std::to_string(iterCount) + "-s" + std::to_string(numselect) + "-f" + 
											std::to_string(frameCount) + ".txt");
						}
						
					}
					
					//每次迭代 都要做下面的事：
					//重新改回非线性优化优化模块
					/*
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
					//保存每个点与对应最近点 的距离   
        			std::vector<float> match_dist;
					std::vector<int> frame_dist;//记录frame-id之间的距离  ,每个列表单独记录
					// fdistThs.clear();
					TicToc t_scansample;
					for (int i = 0; i < int(CloudSampledFeat.size()); i++)
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
					//另一种方式，从前100里去除 外点，for overlap test;所以只遍历前numselect个点
					//统计在得到9*numselect个点中所访问的点个数
					int numvisited = 0;

					// frame_dist.clear();
					numpicked = 0;//计数
					for (int i = 0; i < numProcessed; i++)//遍历所有点 直到找满100个点 numProcessed
					{
						int ind1 = cloudSortInd1[i];//值较大点的索引
						//outlierrm还是xyzi格式
						// pointOri = Outlierm->points[ind1]; 
						// XYZIToXYZINormal(&pointOri, &pointWN);
						pointOri = CloudProcessed->points[ind1];
						pointWNSel = ScanWN->points[ind1];
						// Eigen::Vector3d pointf(pointWN.x, pointWN.y, pointWN.z);
						// pointWNormalAssociateToMap(&pointWN, &pointWNSel);//将该采样点转为map世界坐标系  使用的是odom的结果位姿 这步转换是否需要？
						//在现有地图点中找到距离采样点最近的一个点
						int numkdtreeo = kdtreeFromMap->nearestKSearch(pointWNSel, 1, pointSearchInd, pointSearchSqDis);
						if (numkdtreeo<=0)//没出现过 总能找到一个最近点
						{
							std::cerr<<"Error: No nearest point in map !"<<endl;
							
						}
						//测试不同的outlier rejection:距离，法线夹角
						//source 点的法线
						// Eigen::Vector3d nxi( pointWNSel.normal[0], pointWNSel.normal[1], pointWNSel.normal[2] );
						//对应点的法线
						pointxyzinormal yn = laserCloudFromMap->points[ pointSearchInd[0] ];
						// Eigen::Vector3d nj( yn.normal[0], yn.normal[1], yn.normal[2] );
						// float n_cos = nxi.dot(nj)/(nxi.norm() * nj.norm());
						// float n_degree = acos(n_cos) * 180 / M_PI;//度 夹角 (0,180)

						// if ( sqrt(pointSearchSqDis[0]) > OutlierThr || //此两点之间距离太大，认为是野点
						//     n_degree > n_angleths )//法线一致性，法线夹角太大，是野点
						if (sqrt(pointSearchSqDis[0]) > OutlierThr) 
						{	//打印出最小的距离，用于调整阈值大小
							// std::cout<<"#1 outlier ! mindist="<<sqrt(pointSearchSqDis[0])<<endl;
							// outlierfeat_1 << pointf.x() << " " <<pointf.y() << " " << pointf.z()<< endl;
							pointSearchInd.clear();
                			pointSearchSqDis.clear();
							continue;//跳过，下个点
						}
						//记录对应点的距离
						Eigen::Vector3d xf(pointOri.x, pointOri.y, pointOri.z);
						Eigen::Vector3d yw(yn.x, yn.y, yn.z);
						int yn_id = yn.intensity;
						int xf_id = frameCount;
						int fidist = xf_id - yn_id;
						frame_dist.push_back(fidist);
						// Eigen::Vector3d x_yid = pose_gt[yn_id].inverse() * pose_gt[xf_id] * xf;
						// //得到yn_id对应的历史算法估计的位姿
						// Eigen::Quaterniond q_wy(laserAfterMappedPath.poses[yn_id].pose.orientation.w, 
						// 						laserAfterMappedPath.poses[yn_id].pose.orientation.x, 
						// 						laserAfterMappedPath.poses[yn_id].pose.orientation.y, 
						// 						laserAfterMappedPath.poses[yn_id].pose.orientation.z);
						
						// Eigen::Vector3d t_wy(laserAfterMappedPath.poses[yn_id].pose.position.x, 
						// 					 laserAfterMappedPath.poses[yn_id].pose.position.y, 
						// 					 laserAfterMappedPath.poses[yn_id].pose.position.z);
						// Eigen::Vector3d y_yid = q_wy.inverse() * (yw - t_wy);
						float dd = sqrt(pointSearchSqDis[0]);//(x_yid - y_yid).norm()
						match_dist.push_back(dd);//dd是用真值计算的对应点的距离
						pointWNSel.intensity = fidist;
						if ( iterCount == 0 || iterCount == numicp-1 )//只保存numICP-1次匹配后的点对距离
						{
							//xid yid dd
							// outfeatpair << 1 << " " << yn_id << " " << dd <<endl;
							
						}

						CloudSampledFeat[0].push_back(pointWNSel);
						numpicked++;
						pointSearchInd.clear();
                		pointSearchSqDis.clear();
						if (numpicked >= numselect)//若已经选够100点 结束循环
						{
							numvisited = numvisited + i+1; //统计所有访问到的点的个数
							break;
						}
						if (i == numProcessed-1)
						{
							numvisited = numvisited + i+1; //统计所有访问到的点的个数
						}
							
					}
					/*if(frame_dist.size() != numselect){
						std::cerr<<"SIZE ERROR!"<<endl;
					}
					std::sort(frame_dist.begin(), frame_dist.end(), std::greater<int>());
					int thrind = frame_dist.size() * fdistrmperc - 1;
					int dist_thr = frame_dist[thrind];//x%较大/小的帧距
					if(dist_thr<=1)//帧距最小也是1啊
					{
						dist_thr = 2;
					}
					fdistThs[0] = dist_thr;
					cout<<"percentage "<<fdistrmperc<<" , list 1 dist_thr: "<<dist_thr<<endl;*/

					// frame_dist.clear();
					numpicked = 0;//计数器清零
					for (int i = 0; i < numProcessed; i++)//遍历所有点 直到找满100个点
					{
						int ind2 = cloudSortInd2[i];//值较大点的索引
						pointOri = CloudProcessed->points[ind2];
						pointWNSel = ScanWN->points[ind2];
						// pointWNormalAssociateToMap(&pointWN, &pointWNSel);//将该采样点转为map世界坐标系  使用的是odom的结果位姿
						//在现有地图点中找到距离采样点最近的一个点
						int numkdtreeo = kdtreeFromMap->nearestKSearch(pointWNSel, 1, pointSearchInd, pointSearchSqDis);
						if (numkdtreeo<=0)//没出现过 总能找到一个最近点
						{
							std::cerr<<"Error: No nearest point in map !"<<endl;
							
						}
						//测试不同的outlier rejection:距离，法线夹角
						//source 点的法线
						//Eigen::Vector3d nxi( pointWNSel.normal[0], pointWNSel.normal[1], pointWNSel.normal[2] );
						//对应点的法线
						pointxyzinormal yn = laserCloudFromMap->points[ pointSearchInd[0] ];
						// Eigen::Vector3d nj( yn.normal[0], yn.normal[1], yn.normal[2] );
						// float n_cos = nxi.dot(nj)/(nxi.norm() * nj.norm());
						// float n_degree = acos(n_cos) * 180 / M_PI;//度 夹角 (0,180)

						// if ( sqrt(pointSearchSqDis[0]) > OutlierThr || //此两点之间距离太大，认为是野点
						//     n_degree > n_angleths )//法线一致性，法线夹角太大，是野点
						if (sqrt(pointSearchSqDis[0]) > OutlierThr)
						{
							pointSearchInd.clear();
                			pointSearchSqDis.clear();
							continue;//跳过，下个点
						}
						//记录对应点的距离
						Eigen::Vector3d xf(pointOri.x, pointOri.y, pointOri.z);
						Eigen::Vector3d yw(yn.x, yn.y, yn.z);
						int yn_id = yn.intensity;
						int xf_id = frameCount;
						int fidist = xf_id - yn_id;
						frame_dist.push_back(fidist);
						// Eigen::Vector3d x_yid = pose_gt[yn_id].inverse() * pose_gt[xf_id] * xf;
						// //得到yn_id对应的历史算法估计的位姿
						// Eigen::Quaterniond q_wy(laserAfterMappedPath.poses[yn_id].pose.orientation.w, 
						// 						laserAfterMappedPath.poses[yn_id].pose.orientation.x, 
						// 						laserAfterMappedPath.poses[yn_id].pose.orientation.y, 
						// 						laserAfterMappedPath.poses[yn_id].pose.orientation.z);
						
						// Eigen::Vector3d t_wy(laserAfterMappedPath.poses[yn_id].pose.position.x, 
						// 					 laserAfterMappedPath.poses[yn_id].pose.position.y, 
						// 					 laserAfterMappedPath.poses[yn_id].pose.position.z);
						// Eigen::Vector3d y_yid = q_wy.inverse() * (yw - t_wy);
						float dd = sqrt(pointSearchSqDis[0]);//(x_yid - y_yid).norm()
						match_dist.push_back(dd);
						pointWNSel.intensity = fidist;
						if ( iterCount == 0 || iterCount == numicp-1 )//只保存numICP-1次匹配后的点对距离
						{
							//xid yid dd
							// outfeatpair << 2 << " " << yn_id << " " << dd <<endl;
							
						}
						CloudSampledFeat[1].push_back(pointWNSel);
						numpicked++;
						pointSearchInd.clear();
                		pointSearchSqDis.clear();
						if (numpicked >= numselect)//若已经选够100点 结束循环
						{
							numvisited = numvisited + i+1; //统计所有访问到的点的个数
							break;
						}
						if (i == numProcessed-1)
						{
							numvisited = numvisited + i+1; //统计所有访问到的点的个数
						}
					}
					/*if(frame_dist.size() != numselect){
						std::cerr<<"SIZE ERROR!"<<endl;
					}
					std::sort(frame_dist.begin(), frame_dist.end(), std::greater<int>());
					thrind = frame_dist.size() * fdistrmperc - 1;
					dist_thr = frame_dist[thrind];//x%较大/小的帧距
					if(dist_thr<=1)//帧距最小也是1啊
					{
						dist_thr = 2;
					}
					fdistThs[1] = dist_thr;
					cout<<"percentage "<<fdistrmperc<<" , list 2 dist_thr: "<<dist_thr<<endl;*/

					// frame_dist.clear();
					numpicked = 0;//计数器清零
					for (int i = 0; i < numProcessed; i++)//遍历所有点 直到找满100个点
					{
						int ind3 = cloudSortInd3[i];//值较大点的索引
						pointOri = CloudProcessed->points[ind3]; 
						pointWNSel = ScanWN->points[ind3];
						// pointWNormalAssociateToMap(&pointWN, &pointWNSel);//将该采样点转为map世界坐标系  使用的是odom的结果位姿
						//在现有地图点中找到距离采样点最近的一个点
						int numkdtreeo = kdtreeFromMap->nearestKSearch(pointWNSel, 1, pointSearchInd, pointSearchSqDis);
						if (numkdtreeo<=0)//没出现过 总能找到一个最近点
						{
							std::cerr<<"Error: No nearest point in map !"<<endl;
							
						}
						//测试不同的outlier rejection:距离，法线夹角
						//source 点的法线
						// Eigen::Vector3d nxi( pointWNSel.normal[0], pointWNSel.normal[1], pointWNSel.normal[2] );
						//对应点的法线
						pointxyzinormal yn = laserCloudFromMap->points[ pointSearchInd[0] ];
						// Eigen::Vector3d nj( yn.normal[0], yn.normal[1], yn.normal[2] );
						// float n_cos = nxi.dot(nj)/(nxi.norm() * nj.norm());
						// float n_degree = acos(n_cos) * 180 / M_PI;//度 夹角 (0,180)

						// if ( sqrt(pointSearchSqDis[0]) > OutlierThr || //此两点之间距离太大，认为是野点
						//     n_degree > n_angleths )//法线一致性，法线夹角太大，是野点
						if (sqrt(pointSearchSqDis[0]) > OutlierThr)
						{
							pointSearchInd.clear();
                			pointSearchSqDis.clear();
							continue;//跳过，下个点
						}
						//记录对应点的距离
						Eigen::Vector3d xf(pointOri.x, pointOri.y, pointOri.z);
						Eigen::Vector3d yw(yn.x, yn.y, yn.z);
						int yn_id = yn.intensity;
						int xf_id = frameCount;
						int fidist = xf_id - yn_id;
						frame_dist.push_back(fidist);
						// Eigen::Vector3d x_yid = pose_gt[yn_id].inverse() * pose_gt[xf_id] * xf;
						// //得到yn_id对应的历史算法估计的位姿
						// Eigen::Quaterniond q_wy(laserAfterMappedPath.poses[yn_id].pose.orientation.w, 
						// 						laserAfterMappedPath.poses[yn_id].pose.orientation.x, 
						// 						laserAfterMappedPath.poses[yn_id].pose.orientation.y, 
						// 						laserAfterMappedPath.poses[yn_id].pose.orientation.z);
						
						// Eigen::Vector3d t_wy(laserAfterMappedPath.poses[yn_id].pose.position.x, 
						// 					 laserAfterMappedPath.poses[yn_id].pose.position.y, 
						// 					 laserAfterMappedPath.poses[yn_id].pose.position.z);
						// Eigen::Vector3d y_yid = q_wy.inverse() * (yw - t_wy);
						float dd = sqrt(pointSearchSqDis[0]);//(x_yid - y_yid).norm()
						match_dist.push_back(dd);
						pointWNSel.intensity = fidist;
						if ( iterCount == 0 || iterCount == numicp-1 )//只保存numICP-1次匹配后的点对距离
						{
							//xid yid dd
							// outfeatpair << 3 << " " << yn_id << " " << dd <<endl;
							
						}
						CloudSampledFeat[2].push_back(pointWNSel);
						// outfeat_3 << pointf.x() << " " <<pointf.y() << " " << pointf.z()<< endl;
						numpicked++;
						pointSearchInd.clear();
                		pointSearchSqDis.clear();
						if (numpicked >= numselect)//若已经选够100点 结束循环
						{
							numvisited = numvisited + i+1; //统计所有访问到的点的个数
							break;
						}
						if (i == numProcessed-1)
						{
							numvisited = numvisited + i+1; //统计所有访问到的点的个数
						}
					}
					/*if(frame_dist.size() != numselect){
						std::cerr<<"SIZE ERROR!"<<endl;
					}
					std::sort(frame_dist.begin(), frame_dist.end(), std::greater<int>());
					thrind = frame_dist.size() * fdistrmperc - 1;
					dist_thr = frame_dist[thrind];//x%较大/小的帧距
					if(dist_thr<=1)//帧距最小也是1啊
					{
						dist_thr = 2;
					}
					fdistThs[2] = dist_thr;
					cout<<"percentage "<<fdistrmperc<<" , list 3 dist_thr: "<<dist_thr<<endl;*/

					// frame_dist.clear();
					numpicked = 0;//计数器清零
					for (int i = 0; i < numProcessed; i++)//遍历所有点 直到找满100个点
					{
						int ind4 = cloudSortInd4[i];//值较大点的索引
						pointOri = CloudProcessed->points[ind4]; 
						pointWNSel = ScanWN->points[ind4];
						// pointWNormalAssociateToMap(&pointWN, &pointWNSel);//将该采样点转为map世界坐标系  使用的是odom的结果位姿
						//在现有地图点中找到距离采样点最近的一个点
						int numkdtreeo = kdtreeFromMap->nearestKSearch(pointWNSel, 1, pointSearchInd, pointSearchSqDis);
						if (numkdtreeo<=0)//没出现过 总能找到一个最近点
						{
							std::cerr<<"Error: No nearest point in map !"<<endl;
							
						}
						//测试不同的outlier rejection:距离，法线夹角
						//source 点的法线
						// Eigen::Vector3d nxi( pointWNSel.normal[0], pointWNSel.normal[1], pointWNSel.normal[2] );
						//对应点的法线
						pointxyzinormal yn = laserCloudFromMap->points[ pointSearchInd[0] ];
						// Eigen::Vector3d nj( yn.normal[0], yn.normal[1], yn.normal[2] );
						// float n_cos = nxi.dot(nj)/(nxi.norm() * nj.norm());
						// float n_degree = acos(n_cos) * 180 / M_PI;//度 夹角 (0,180)

						// if ( sqrt(pointSearchSqDis[0]) > OutlierThr || //此两点之间距离太大，认为是野点
						//     n_degree > n_angleths )//法线一致性，法线夹角太大，是野点
						if (sqrt(pointSearchSqDis[0]) > OutlierThr)
						{
							pointSearchInd.clear();
                			pointSearchSqDis.clear();
							continue;//跳过，下个点
						}
						//记录对应点的距离
						Eigen::Vector3d xf(pointOri.x, pointOri.y, pointOri.z);
						Eigen::Vector3d yw(yn.x, yn.y, yn.z);
						int yn_id = yn.intensity;
						int xf_id = frameCount;
						int fidist = xf_id - yn_id;
						frame_dist.push_back(fidist);
						// Eigen::Vector3d x_yid = pose_gt[yn_id].inverse() * pose_gt[xf_id] * xf;
						// //得到yn_id对应的历史算法估计的位姿
						// Eigen::Quaterniond q_wy(laserAfterMappedPath.poses[yn_id].pose.orientation.w, 
						// 						laserAfterMappedPath.poses[yn_id].pose.orientation.x, 
						// 						laserAfterMappedPath.poses[yn_id].pose.orientation.y, 
						// 						laserAfterMappedPath.poses[yn_id].pose.orientation.z);
						
						// Eigen::Vector3d t_wy(laserAfterMappedPath.poses[yn_id].pose.position.x, 
						// 					 laserAfterMappedPath.poses[yn_id].pose.position.y, 
						// 					 laserAfterMappedPath.poses[yn_id].pose.position.z);
						// Eigen::Vector3d y_yid = q_wy.inverse() * (yw - t_wy);
						float dd = sqrt(pointSearchSqDis[0]);//(x_yid - y_yid).norm()
						match_dist.push_back(dd);
						pointWNSel.intensity = fidist;
						if ( iterCount == 0 || iterCount == numicp-1 )//只保存numICP-1次匹配后的点对距离
						{
							//xid yid dd
							// outfeatpair << 4 << " " << yn_id << " " << dd <<endl;
							
						}
						CloudSampledFeat[3].push_back(pointWNSel);
						numpicked++;
						pointSearchInd.clear();
                		pointSearchSqDis.clear();
						if (numpicked >= numselect)//若已经选够100点 结束循环
						{
							numvisited = numvisited + i+1; //统计所有访问到的点的个数
							break;
						}
						if (i == numProcessed-1)
						{
							numvisited = numvisited + i+1; //统计所有访问到的点的个数
						}
					}
					/*if(frame_dist.size() != numselect){
						std::cerr<<"SIZE ERROR!"<<endl;
					}
					std::sort(frame_dist.begin(), frame_dist.end(), std::greater<int>());
					thrind = frame_dist.size() * fdistrmperc - 1;
					dist_thr = frame_dist[thrind];//x%较大/小的帧距
					if(dist_thr<=1)//帧距最小也是1啊
					{
						dist_thr = 2;
					}
					fdistThs[3] = dist_thr;
					cout<<"percentage "<<fdistrmperc<<" , list 4 dist_thr: "<<dist_thr<<endl;*/

					// frame_dist.clear();
					numpicked = 0;//计数器清零
					for (int i = 0; i < numProcessed; i++)//遍历所有点 直到找满100个点
					{
						int ind5 = cloudSortInd5[i];//值较大点的索引
						pointOri = CloudProcessed->points[ind5]; 
						pointWNSel = ScanWN->points[ind5];
						// pointWNormalAssociateToMap(&pointWN, &pointWNSel);//将该采样点转为map世界坐标系  使用的是odom的结果位姿
						//在现有地图点中找到距离采样点最近的一个点
						int numkdtreeo = kdtreeFromMap->nearestKSearch(pointWNSel, 1, pointSearchInd, pointSearchSqDis);
						if (numkdtreeo<=0)//没出现过 总能找到一个最近点
						{
							std::cerr<<"Error: No nearest point in map !"<<endl;
							
						}
						//测试不同的outlier rejection:距离，法线夹角
						//source 点的法线
						// Eigen::Vector3d nxi( pointWNSel.normal[0], pointWNSel.normal[1], pointWNSel.normal[2] );
						//对应点的法线
						pointxyzinormal yn = laserCloudFromMap->points[ pointSearchInd[0] ];
						// Eigen::Vector3d nj( yn.normal[0], yn.normal[1], yn.normal[2] );
						// float n_cos = nxi.dot(nj)/(nxi.norm() * nj.norm());
						// float n_degree = acos(n_cos) * 180 / M_PI;//度 夹角 (0,180)

						// if ( sqrt(pointSearchSqDis[0]) > OutlierThr || //此两点之间距离太大，认为是野点
						//     n_degree > n_angleths )//法线一致性，法线夹角太大，是野点
						if (sqrt(pointSearchSqDis[0]) > OutlierThr)
						{
							pointSearchInd.clear();
                			pointSearchSqDis.clear();
							continue;//跳过，下个点
						}
						//记录对应点的距离
						Eigen::Vector3d xf(pointOri.x, pointOri.y, pointOri.z);
						Eigen::Vector3d yw(yn.x, yn.y, yn.z);
						int yn_id = yn.intensity;
						int xf_id = frameCount;
						int fidist = xf_id - yn_id;
						frame_dist.push_back(fidist);
						// Eigen::Vector3d x_yid = pose_gt[yn_id].inverse() * pose_gt[xf_id] * xf;
						// //得到yn_id对应的历史算法估计的位姿
						// Eigen::Quaterniond q_wy(laserAfterMappedPath.poses[yn_id].pose.orientation.w, 
						// 						laserAfterMappedPath.poses[yn_id].pose.orientation.x, 
						// 						laserAfterMappedPath.poses[yn_id].pose.orientation.y, 
						// 						laserAfterMappedPath.poses[yn_id].pose.orientation.z);
						
						// Eigen::Vector3d t_wy(laserAfterMappedPath.poses[yn_id].pose.position.x, 
						// 					 laserAfterMappedPath.poses[yn_id].pose.position.y, 
						// 					 laserAfterMappedPath.poses[yn_id].pose.position.z);
						// Eigen::Vector3d y_yid = q_wy.inverse() * (yw - t_wy);
						float dd = sqrt(pointSearchSqDis[0]);//(x_yid - y_yid).norm()
						match_dist.push_back(dd);
						pointWNSel.intensity = fidist;
						if ( iterCount == 0 || iterCount == numicp-1 )//只保存numICP-1次匹配后的点对距离
						{
							//xid yid dd
							// outfeatpair << 5 << " " << yn_id << " " << dd <<endl;
							
						}
						CloudSampledFeat[4].push_back(pointWNSel);
						numpicked++;
						pointSearchInd.clear();
                		pointSearchSqDis.clear();
						if (numpicked >= numselect)//若已经选够100点 结束循环
						{
							numvisited = numvisited + i+1; //统计所有访问到的点的个数
							break;
						}
						if (i == numProcessed-1)
						{
							numvisited = numvisited + i+1; //统计所有访问到的点的个数
						}
					}
					/*if(frame_dist.size() != numselect){
						std::cerr<<"SIZE ERROR!"<<endl;
					}
					std::sort(frame_dist.begin(), frame_dist.end(), std::greater<int>());
					thrind = frame_dist.size() * fdistrmperc - 1;
					dist_thr = frame_dist[thrind];//x%较大/小的帧距
					if(dist_thr<=1)//帧距最小也是1啊
					{
						dist_thr = 2;
					}
					fdistThs[4] = dist_thr;
					cout<<"percentage "<<fdistrmperc<<" , list 5 dist_thr: "<<dist_thr<<endl;*/

					// frame_dist.clear();
					numpicked = 0;//计数器清零
					for (int i = 0; i < numProcessed; i++)//遍历所有点 直到找满100个点
					{
						int ind6 = cloudSortInd6[i];//值较大点的索引
						pointOri = CloudProcessed->points[ind6]; 
						pointWNSel = ScanWN->points[ind6];
						// pointWNormalAssociateToMap(&pointWN, &pointWNSel);//将该采样点转为map世界坐标系  使用的是odom的结果位姿
						//在现有地图点中找到距离采样点最近的一个点
						int numkdtreeo = kdtreeFromMap->nearestKSearch(pointWNSel, 1, pointSearchInd, pointSearchSqDis);
						if (numkdtreeo<=0)//没出现过 总能找到一个最近点
						{
							std::cerr<<"Error: No nearest point in map !"<<endl;
							
						}
						//测试不同的outlier rejection:距离，法线夹角
						//source 点的法线
						// Eigen::Vector3d nxi( pointWNSel.normal[0], pointWNSel.normal[1], pointWNSel.normal[2] );
						//对应点的法线
						pointxyzinormal yn = laserCloudFromMap->points[ pointSearchInd[0] ];
						// Eigen::Vector3d nj( yn.normal[0], yn.normal[1], yn.normal[2] );
						// float n_cos = nxi.dot(nj)/(nxi.norm() * nj.norm());
						// float n_degree = acos(n_cos) * 180 / M_PI;//度 夹角 (0,180)

						// if ( sqrt(pointSearchSqDis[0]) > OutlierThr || //此两点之间距离太大，认为是野点
						//     n_degree > n_angleths )//法线一致性，法线夹角太大，是野点
						if (sqrt(pointSearchSqDis[0]) > OutlierThr)
						{
							pointSearchInd.clear();
                			pointSearchSqDis.clear();
							continue;//跳过，下个点
						}
						//记录对应点的距离
						Eigen::Vector3d xf(pointOri.x, pointOri.y, pointOri.z);
						Eigen::Vector3d yw(yn.x, yn.y, yn.z);
						int yn_id = yn.intensity;
						int xf_id = frameCount;
						int fidist = xf_id - yn_id;
						frame_dist.push_back(fidist);
						// Eigen::Vector3d x_yid = pose_gt[yn_id].inverse() * pose_gt[xf_id] * xf;
						// //得到yn_id对应的历史算法估计的位姿
						// Eigen::Quaterniond q_wy(laserAfterMappedPath.poses[yn_id].pose.orientation.w, 
						// 						laserAfterMappedPath.poses[yn_id].pose.orientation.x, 
						// 						laserAfterMappedPath.poses[yn_id].pose.orientation.y, 
						// 						laserAfterMappedPath.poses[yn_id].pose.orientation.z);
						
						// Eigen::Vector3d t_wy(laserAfterMappedPath.poses[yn_id].pose.position.x, 
						// 					 laserAfterMappedPath.poses[yn_id].pose.position.y, 
						// 					 laserAfterMappedPath.poses[yn_id].pose.position.z);
						// Eigen::Vector3d y_yid = q_wy.inverse() * (yw - t_wy);
						float dd = sqrt(pointSearchSqDis[0]);//(x_yid - y_yid).norm()
						match_dist.push_back(dd);
						pointWNSel.intensity = fidist;
						if ( iterCount == 0 || iterCount == numicp-1 )//只保存numICP-1次匹配后的点对距离
						{
							//xid yid dd
							// outfeatpair << 6 << " " << yn_id << " " << dd <<endl;
							
						}
						CloudSampledFeat[5].push_back(pointWNSel);
						numpicked++;
						pointSearchInd.clear();
                		pointSearchSqDis.clear();
						if (numpicked >= numselect)//若已经选够100点 结束循环
						{
							numvisited = numvisited + i+1; //统计所有访问到的点的个数
							break;
						}
						if (i == numProcessed-1)
						{
							numvisited = numvisited + i+1; //统计所有访问到的点的个数
						}
					}
					/*if(frame_dist.size() != numselect){
						std::cerr<<"SIZE ERROR!"<<endl;
					}
					std::sort(frame_dist.begin(), frame_dist.end(), std::greater<int>());
					thrind = frame_dist.size() * fdistrmperc - 1;
					dist_thr = frame_dist[thrind];//x%较大/小的帧距
					if(dist_thr<=1)//帧距最小也是1啊
					{
						dist_thr = 2;
					}
					fdistThs[5] = dist_thr;
					cout<<"percentage "<<fdistrmperc<<" , list 6 dist_thr: "<<dist_thr<<endl;*/

					// frame_dist.clear();
					numpicked = 0;//计数器清零
					for (int i = 0; i < numProcessed; i++)//遍历所有点 直到找满100个点
					{
						int ind7 = cloudSortInd7[i];//值较大点的索引
						pointOri = CloudProcessed->points[ind7]; 
						pointWNSel = ScanWN->points[ind7];
						// pointWNormalAssociateToMap(&pointWN, &pointWNSel);//将该采样点转为map世界坐标系  使用的是odom的结果位姿
						//在现有地图点中找到距离采样点最近的一个点
						int numkdtreeo = kdtreeFromMap->nearestKSearch(pointWNSel, 1, pointSearchInd, pointSearchSqDis);
						if (numkdtreeo<=0)//没出现过 总能找到一个最近点
						{
							std::cerr<<"Error: No nearest point in map !"<<endl;
							
						}
						//测试不同的outlier rejection:距离，法线夹角
						//source 点的法线
						// Eigen::Vector3d nxi( pointWNSel.normal[0], pointWNSel.normal[1], pointWNSel.normal[2] );
						//对应点的法线
						pointxyzinormal yn = laserCloudFromMap->points[ pointSearchInd[0] ];
						// Eigen::Vector3d nj( yn.normal[0], yn.normal[1], yn.normal[2] );
						// float n_cos = nxi.dot(nj)/(nxi.norm() * nj.norm());
						// float n_degree = acos(n_cos) * 180 / M_PI;//度 夹角 (0,180)

						// if ( sqrt(pointSearchSqDis[0]) > OutlierThr || //此两点之间距离太大，认为是野点
						//     n_degree > n_angleths )//法线一致性，法线夹角太大，是野点
						if (sqrt(pointSearchSqDis[0]) > OutlierThr)
						{
							pointSearchInd.clear();
                			pointSearchSqDis.clear();
							continue;//跳过，下个点
						}
						//记录对应点的距离
						Eigen::Vector3d xf(pointOri.x, pointOri.y, pointOri.z);
						Eigen::Vector3d yw(yn.x, yn.y, yn.z);
						int yn_id = yn.intensity;
						int xf_id = frameCount;
						int fidist = xf_id - yn_id;
						frame_dist.push_back(fidist);
						// Eigen::Vector3d x_yid = pose_gt[yn_id].inverse() * pose_gt[xf_id] * xf;
						// //得到yn_id对应的历史算法估计的位姿
						// Eigen::Quaterniond q_wy(laserAfterMappedPath.poses[yn_id].pose.orientation.w, 
						// 						laserAfterMappedPath.poses[yn_id].pose.orientation.x, 
						// 						laserAfterMappedPath.poses[yn_id].pose.orientation.y, 
						// 						laserAfterMappedPath.poses[yn_id].pose.orientation.z);
						
						// Eigen::Vector3d t_wy(laserAfterMappedPath.poses[yn_id].pose.position.x, 
						// 					 laserAfterMappedPath.poses[yn_id].pose.position.y, 
						// 					 laserAfterMappedPath.poses[yn_id].pose.position.z);
						// Eigen::Vector3d y_yid = q_wy.inverse() * (yw - t_wy);
						float dd = sqrt(pointSearchSqDis[0]);//(x_yid - y_yid).norm()
						match_dist.push_back(dd);
						pointWNSel.intensity = fidist;
						if ( iterCount == 0 || iterCount == numicp-1 )//只保存numICP-1次匹配后的点对距离
						{
							//xid yid dd
							// outfeatpair << 7 << " " << yn_id << " " << dd <<endl;
							
						}
						CloudSampledFeat[6].push_back(pointWNSel);
						numpicked++;
						pointSearchInd.clear();
                		pointSearchSqDis.clear();
						if (numpicked >= numselect)//若已经选够100点 结束循环
						{
							numvisited = numvisited + i+1; //统计所有访问到的点的个数
							break;
						}
						if (i == numProcessed-1)
						{
							numvisited = numvisited + i+1; //统计所有访问到的点的个数
						}
					}
					/*if(frame_dist.size() != numselect){
						std::cerr<<"SIZE ERROR!"<<endl;
					}
					std::sort(frame_dist.begin(), frame_dist.end(), std::greater<int>());
					thrind = frame_dist.size() * fdistrmperc - 1;
					dist_thr = frame_dist[thrind];//x%较大/小的帧距
					if(dist_thr<=1)//帧距最小也是1啊
					{
						dist_thr = 2;
					}
					fdistThs[6] = dist_thr;
					cout<<"percentage "<<fdistrmperc<<" , list 7 dist_thr: "<<dist_thr<<endl;*/

					// frame_dist.clear();
					numpicked = 0;//计数器清零
					for (int i = 0; i < numProcessed; i++)//遍历所有点 直到找满100个点
					{
						int ind8 = cloudSortInd8[i];//值较大点的索引
						pointOri = CloudProcessed->points[ind8]; 
						pointWNSel = ScanWN->points[ind8];
						// pointWNormalAssociateToMap(&pointWN, &pointWNSel);//将该采样点转为map世界坐标系  使用的是odom的结果位姿
						//在现有地图点中找到距离采样点最近的一个点
						int numkdtreeo = kdtreeFromMap->nearestKSearch(pointWNSel, 1, pointSearchInd, pointSearchSqDis);
						if (numkdtreeo<=0)//没出现过 总能找到一个最近点
						{
							std::cerr<<"Error: No nearest point in map !"<<endl;
							
						}
						//测试不同的outlier rejection:距离，法线夹角
						//source 点的法线
						// Eigen::Vector3d nxi( pointWNSel.normal[0], pointWNSel.normal[1], pointWNSel.normal[2] );
						//对应点的法线
						pointxyzinormal yn = laserCloudFromMap->points[ pointSearchInd[0] ];
						// Eigen::Vector3d nj( yn.normal[0], yn.normal[1], yn.normal[2] );
						// float n_cos = nxi.dot(nj)/(nxi.norm() * nj.norm());
						// float n_degree = acos(n_cos) * 180 / M_PI;//度 夹角 (0,180)

						// if ( sqrt(pointSearchSqDis[0]) > OutlierThr || //此两点之间距离太大，认为是野点
						//     n_degree > n_angleths )//法线一致性，法线夹角太大，是野点
						if (sqrt(pointSearchSqDis[0]) > OutlierThr) 
						{
							pointSearchInd.clear();
                			pointSearchSqDis.clear();
							continue;//跳过，下个点
						}
						//记录对应点的距离
						Eigen::Vector3d xf(pointOri.x, pointOri.y, pointOri.z);
						Eigen::Vector3d yw(yn.x, yn.y, yn.z);
						int yn_id = yn.intensity;
						int xf_id = frameCount;
						int fidist = xf_id - yn_id;
						frame_dist.push_back(fidist);
						// Eigen::Vector3d x_yid = pose_gt[yn_id].inverse() * pose_gt[xf_id] * xf;
						// //得到yn_id对应的历史算法估计的位姿
						// Eigen::Quaterniond q_wy(laserAfterMappedPath.poses[yn_id].pose.orientation.w, 
						// 						laserAfterMappedPath.poses[yn_id].pose.orientation.x, 
						// 						laserAfterMappedPath.poses[yn_id].pose.orientation.y, 
						// 						laserAfterMappedPath.poses[yn_id].pose.orientation.z);
						
						// Eigen::Vector3d t_wy(laserAfterMappedPath.poses[yn_id].pose.position.x, 
						// 					 laserAfterMappedPath.poses[yn_id].pose.position.y, 
						// 					 laserAfterMappedPath.poses[yn_id].pose.position.z);
						// Eigen::Vector3d y_yid = q_wy.inverse() * (yw - t_wy);
						float dd = sqrt(pointSearchSqDis[0]);//(x_yid - y_yid).norm()
						match_dist.push_back(dd);
						pointWNSel.intensity = fidist;
						if ( iterCount == 0 || iterCount == numicp-1 )//只保存numICP-1次匹配后的点对距离
						{
							//xid yid dd
							// outfeatpair << 8 << " " << yn_id << " " << dd <<endl;
							
						}
						CloudSampledFeat[7].push_back(pointWNSel);
						numpicked++;
						pointSearchInd.clear();
                		pointSearchSqDis.clear();
						if (numpicked >= numselect)//若已经选够100点 结束循环
						{
							numvisited = numvisited + i+1; //统计所有访问到的点的个数
							break;
						}
						if (i == numProcessed-1)
						{
							numvisited = numvisited + i+1; //统计所有访问到的点的个数
						}
					}
					/*if(frame_dist.size() != numselect){
						std::cerr<<"SIZE ERROR!"<<endl;
					}
					std::sort(frame_dist.begin(), frame_dist.end(), std::greater<int>());
					thrind = frame_dist.size() * fdistrmperc - 1;
					dist_thr = frame_dist[thrind];//x%较大/小的帧距
					if(dist_thr<=1)//帧距最小也是1啊
					{
						dist_thr = 2;
					}
					fdistThs[7] = dist_thr;
					cout<<"percentage "<<fdistrmperc<<" , list 8 dist_thr: "<<dist_thr<<endl;*/

					// frame_dist.clear();
					numpicked = 0;//计数器清零
					for (int i = 0; i < numProcessed; i++)//遍历所有点 直到找满100个点
					{
						int ind9 = cloudSortInd9[i];//值较大点的索引
						pointOri = CloudProcessed->points[ind9]; 
						pointWNSel = ScanWN->points[ind9];
						// pointWNormalAssociateToMap(&pointWN, &pointWNSel);//将该采样点转为map世界坐标系  使用的是odom的结果位姿
						//在现有地图点中找到距离采样点最近的一个点
						int numkdtreeo = kdtreeFromMap->nearestKSearch(pointWNSel, 1, pointSearchInd, pointSearchSqDis);
						if (numkdtreeo<=0)//没出现过 总能找到一个最近点
						{
							std::cerr<<"Error: No nearest point in map !"<<endl;
							
						}
						//测试不同的outlier rejection:距离，法线夹角
						//source 点的法线
						// Eigen::Vector3d nxi( pointWNSel.normal[0], pointWNSel.normal[1], pointWNSel.normal[2] );
						//对应点的法线
						pointxyzinormal yn = laserCloudFromMap->points[ pointSearchInd[0] ];
						// Eigen::Vector3d nj( yn.normal[0], yn.normal[1], yn.normal[2] );
						// float n_cos = nxi.dot(nj)/(nxi.norm() * nj.norm());
						// float n_degree = acos(n_cos) * 180 / M_PI;//度 夹角 (0,180)

						// if ( sqrt(pointSearchSqDis[0]) > OutlierThr || //此两点之间距离太大，认为是野点
						//     n_degree > n_angleths )//法线一致性，法线夹角太大，是野点
						if (sqrt(pointSearchSqDis[0]) > OutlierThr)
						{
							pointSearchInd.clear();
                			pointSearchSqDis.clear();
							continue;//跳过，下个点
						}
						//记录对应点的距离
						Eigen::Vector3d xf(pointOri.x, pointOri.y, pointOri.z);
						Eigen::Vector3d yw(yn.x, yn.y, yn.z);
						int yn_id = yn.intensity;
						int xf_id = frameCount;
						int fidist = xf_id - yn_id;
						frame_dist.push_back(fidist);
						// Eigen::Vector3d x_yid = pose_gt[yn_id].inverse() * pose_gt[xf_id] * xf;
						// //得到yn_id对应的历史算法估计的位姿
						// Eigen::Quaterniond q_wy(laserAfterMappedPath.poses[yn_id].pose.orientation.w, 
						// 						laserAfterMappedPath.poses[yn_id].pose.orientation.x, 
						// 						laserAfterMappedPath.poses[yn_id].pose.orientation.y, 
						// 						laserAfterMappedPath.poses[yn_id].pose.orientation.z);
						
						// Eigen::Vector3d t_wy(laserAfterMappedPath.poses[yn_id].pose.position.x, 
						// 					 laserAfterMappedPath.poses[yn_id].pose.position.y, 
						// 					 laserAfterMappedPath.poses[yn_id].pose.position.z);
						// Eigen::Vector3d y_yid = q_wy.inverse() * (yw - t_wy);
						float dd = sqrt(pointSearchSqDis[0]);//(x_yid - y_yid).norm()
						match_dist.push_back(dd);
						pointWNSel.intensity = fidist;
						if ( iterCount == 0 || iterCount == numicp-1 )//只保存numICP-1次匹配后的点对距离
						{
							//xid yid dd
							// outfeatpair << 9 << " " << yn_id << " " << dd <<endl;
							
						}
						CloudSampledFeat[8].push_back(pointWNSel);
						numpicked++;
						pointSearchInd.clear();
                		pointSearchSqDis.clear();
						if (numpicked >= numselect)//若已经选够100点 结束循环
						{
							numvisited = numvisited + i+1; //统计所有访问到的点的个数
							break;
						}
						if (i == numProcessed-1)
						{
							numvisited = numvisited + i+1; //统计所有访问到的点的个数
						}
					}
					/*if(frame_dist.size() != numselect){
						std::cerr<<"SIZE ERROR!"<<endl;
					}
					std::sort(frame_dist.begin(), frame_dist.end(), std::greater<int>());
					thrind = frame_dist.size() * fdistrmperc - 1;
					dist_thr = frame_dist[thrind];//x%较大/小的帧距
					if(dist_thr<=1)//帧距最小也是1啊
					{
						dist_thr = 2;
					}
					fdistThs[8] = dist_thr;
					cout<<"percentage "<<fdistrmperc<<" , list 9 dist_thr: "<<dist_thr<<endl;
					*/

					printf("scan sampling time %f ms \n", t_scansample.toc());//采样总时间
					
					if(match_dist.size()!=9*numselect){
						std::cerr<<"SIZE ERROR!"<<endl;
					}
					if(frame_dist.size()!=9*numselect){
						std::cerr<<"SIZE ERROR!"<<endl;
					}
					// if(fdistThs.size()!=9){
					// 	std::cerr<<"Ths SIZE ERROR!"<<endl;
					// }
					if(0)//iterCount >= 2先进行所有点优化2次，之后的18次都要进行滤除匹配较差的一部分点 但效果不好
					{
						std::sort(match_dist.begin(), match_dist.end(), std::greater<float>());//降序排列
						//得到最大的20% 对应的距离 作为阈值
						int thrind = match_dist.size() * matchrmperc - 1; //0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9
						float match_thr = match_dist[thrind];
						cout<<"percentage "<<matchrmperc<<" ,thrind: "<<thrind<<", threshold: "<<match_thr<<endl;
						for (int i = 0; i < int(CloudSampledFeat.size()); i++)
						{
							for (size_t j = 0; j < CloudSampledFeat[i].points.size(); j++)
							{
								pointWN = CloudSampledFeat[i].points[j];
								if (pointWN.intensity < match_thr)//该点对距离值小于 20%阈值 保留
								{
									// pointWN.intensity = i+1;//记录特征列表名字
									CloudSampled->push_back(pointWN);
								}
								
							}
						}
						
					}
					else if (0)//根据帧的远近来过滤 较劲 或 较远的点
					{
						// 用标准库比较函数对象排序  降序greater/ 升序less
						std::sort(frame_dist.begin(), frame_dist.end(), std::greater<int>());
						int thrind = frame_dist.size() * fdistrmperc - 1;
						fdist_thr = frame_dist[thrind];//x%较大/小的帧距
						if(fdist_thr<=1)//帧距最小也是1啊
						{
							fdist_thr = 2;
						}
						cout<<"percentage "<<fdistrmperc<<" ,thrind: "<<thrind<<", fdist_thr: "<<fdist_thr<<endl;
						for (int i = 0; i < int(CloudSampledFeat.size()); i++)
						{
							for (size_t j = 0; j < CloudSampledFeat[i].points.size(); j++)
							{
								pointWN = CloudSampledFeat[i].points[j];
								if (pointWN.intensity < fdist_thr)//近的:该点对帧距不大于 阈值 保留 远的: 不小于   || i == 1
								{
									CloudSampled->push_back(pointWN);
								}
								// if (pointWN.intensity < fdistThs[i])//每个列表阈值不同
								// {
								// 	CloudSampled->push_back(pointWN);
								// }
								
							}
						}
					}
					else
					{//直接全部保留
						for (int i = 0; i < int(CloudSampledFeat.size()); i++)
						{
							*CloudSampled += CloudSampledFeat[i];//将所有列表采样点组合在一块
						}

					}
					
					int numscansampled = CloudSampled->points.size();// overlap test 小于等于900
					std::cout << "the size before sampled : " << numProcessed << " and the size after sampled is " << numscansampled << '\n';
					
					std::vector<double> I_buffer;//记录所有特征点对应的到隐式平面的距离
					for (int i = 0; i < numscansampled; i++)
					{
						pointWNSel = CloudSampled->points[i];//已经是变换后的 有新的法线
						int numkdtree = kdtreeFromMap->radiusSearch(pointWNSel, RadThr, pointSearchInd, pointSearchSqDis);
						if (numkdtree <= 0)//
						{
							pointSearchInd.clear();
                			pointSearchSqDis.clear();
							continue;
						}
						int numBox = pointSearchInd.size();//得到的指定半径内近邻个数
						std::vector<double> Wx;//权值数组
						std::vector<double> wD;//分子中的一项 weighted distance
						Wx.clear();
						wD.clear();
						//pointsel 
						Eigen::Vector3d pointx(pointWNSel.x, pointWNSel.y, pointWNSel.z);
						for (int j = 0; j < numBox; j++) //只计算那个最近点 这时就退化为经典的point-planeICP numBox
						{	//当前来自地图中的点p_
							Eigen::Vector3d pcurrbox(laserCloudFromMap->points[ pointSearchInd[j] ].x,
													laserCloudFromMap->points[ pointSearchInd[j] ].y,
													laserCloudFromMap->points[ pointSearchInd[j] ].z);
							//当前来自地图中的点p_的法线
							// Eigen::Vector3d normpcurr = nj; //近似 都用最近点的法线 test: 是有效的！
							Eigen::Vector3d normpcurr(laserCloudFromMap->points[ pointSearchInd[0] ].normal[0],
													  laserCloudFromMap->points[ pointSearchInd[0] ].normal[1],
													  laserCloudFromMap->points[ pointSearchInd[0] ].normal[2]);
							//统一法线视点! 这里视点使用Viewpoint to test:  pointx(不对!)
							if( normpcurr.dot(pcurrbox - Viewpoint) > 0)
							{//设置法线方向统一
								normpcurr = -normpcurr;
							}
							//当前点对应的权值
							double w_j = exp(-pointSearchSqDis[j]/(h_imls*h_imls));
							//加权法线方向距离
							double tmp1 = w_j*((pointx-pcurrbox).dot(normpcurr));
							Wx.push_back(w_j);
							wD.push_back(tmp1);
						}

						//计算采样点pointx到map点隐式平面的距离
						double fenzi = double(std::accumulate(wD.begin(), wD.end(), 0.0000000));//出现了负值？合理吗 0.000001
						double fenmu = double(std::accumulate(Wx.begin(), Wx.end(), 0.0000000));
						double I_xi = (double)fenzi/fenmu;//会出现NaN
						double Iabs = fabs(I_xi);
						I_buffer.push_back(Iabs);//保存该距离值  按绝对值排序试试 fabs()
						CloudSampled->points[i].intensity = I_xi;

					}
					if(IRmSome && iterCount>9)//是否需要排序
					{
						// 用标准库比较函数对象排序  降序greater/ 升序less 降序排列获得对应的分位数阈值
						std::sort(I_buffer.begin(), I_buffer.end(), std::greater<double>());
						int thrind = I_buffer.size() * I_rmperc - 1;
						Idist_thr = I_buffer[thrind];//x%较大/小的距离
						cout<<"percentage "<<I_rmperc<<" ,thrind: "<<thrind<<", I_thr: "<<Idist_thr<<endl;
						/*if(iterCount==0)
						{
							for (size_t i_ = 0; i_ < I_buffer.size(); i_++)
							{//debug 输出看I的排序
								cout<<I_buffer[i_]<<endl;
							}
						}*/
						
					}
					
					//初始化A，b，x 使用动态大小矩阵
					Eigen::MatrixXf A(numscansampled, 6); 
					Eigen::Matrix< float, 6, 1 > x;
					Eigen::MatrixXf b(numscansampled, 1);
					Eigen::MatrixXf loss(numscansampled, 1);//目标函数loss
					
					//遍历每个采样点x，对应的投影点y 和法线//其实就是找每个点的对应点
					int numResidual = 0;//记录每次优化被加入残差项的点的个数 即实际参与优化的点数，就是A的数据长度
					double sumerror = 0;
					for (int i = 0; i < numscansampled; i++)
					{
						pointWNSel = CloudSampled->points[i];//已经是变换后的 有新的法线
						// pointWNormalAssociateToMap(&pointWN, &pointWNSel);//将该特征点进行变换  使用的是当前？位姿 这步转换是否需要？如何同步地更新优化后的q_w_curr
						//在现有地图点中找到距离采样点不大于0.20m 的点
						int numkdtree = kdtreeFromMap->radiusSearch(pointWNSel, RadThr, pointSearchInd, pointSearchSqDis);
						if (numkdtree <= 0)//
						{
							// std::cout<<"WARNING: 0 BOX point found!! skip this point"<<endl;
							// std::cout<<pointSearchInd[0]<<endl;
							pointSearchInd.clear();
                			pointSearchSqDis.clear();
							continue;
						}
						double I_xi = pointWNSel.intensity;//已经计算好的隐式距离
						if(IRmSome && iterCount>9)//测试 只从3步之后才去除一部分点
						{
							if(fabs(I_xi) >= Idist_thr)
							{//距离大于阈值，跳过
								continue;
							}
						}

						Eigen::Vector3d pointx(pointWNSel.x, pointWNSel.y, pointWNSel.z);
						//source 点的法线
						Eigen::Vector3d nxi( pointWNSel.normal[0], pointWNSel.normal[1], pointWNSel.normal[2] );

						//最近点 来自地图中
						pointxyzinormal nearestp = laserCloudFromMap->points[ pointSearchInd[0] ];
						//最近点的法线直接从数据结构中拿出
						Eigen::Vector3d nj( nearestp.normal[0], nearestp.normal[1], nearestp.normal[2] );
						Eigen::Vector3d nearestv(nearestp.x, nearestp.y, nearestp.z);

						//统一对两个法线进行视点统一 为世界坐标系原点
						if( nxi.dot(pointx - Viewpoint) > 0)
        				{//设置法线方向统一
            				nxi = -nxi;
        				}
						if( nj.dot(nearestv - Viewpoint) > 0)
        				{//设置法线方向统一
            				nj = -nj;
        				}
						
						//根据对应点法线的夹角来进行选择
						float n_cos = nxi.dot(nj)/(nxi.norm() * nj.norm());
						float n_degree = acos(n_cos) * 180 / M_PI;//度 夹角 (0,180)
						if ( iterCount == numicp-1 )//只保存numICP-1次匹配后的点对距离
						{
							//记录角度 隐式距离， 帧距
							outangle << n_degree << " " << I_xi << " " << (frameCount - nearestp.intensity) <<endl;	
						}
						// if (n_degree>n_angleths)
						// {//夹角太大 认为是outlier
						// 	pointSearchInd.clear();
                		// 	pointSearchSqDis.clear();
						// 	continue;
						// }

						//x_i对应的点y_i nearestv
						Eigen::Vector3d y_i = pointx - I_xi * nj;

						// ceres::CostFunction *cost_function = LidarPoint2PlaneICP::Create(x_i, y_i, nj); //本质上优化的是I_xi
						// problem.AddResidualBlock(cost_function, loss_function, parameters, parameters + 4);
						
						sumerror = sumerror + nj.dot(y_i-pointx) * nj.dot(y_i-pointx);

						//分别对A，b 赋值
						b(numResidual, 0) = nj.dot(y_i-pointx);//
						
						A(numResidual, 0) = nj.z() * pointx.y() - nj.y() * pointx.z();
						A(numResidual, 1) = nj.x() * pointx.z() - nj.z() * pointx.x();
						A(numResidual, 2) = nj.y() * pointx.x() - nj.x() * pointx.y();
						A(numResidual, 3) = nj.x();
						A(numResidual, 4) = nj.y();
						A(numResidual, 5) = nj.z();
						//统计所有matched point数目
						numResidual = numResidual + 1;
						//clear kdtree return vector
						pointSearchInd.clear();
						pointSearchSqDis.clear();
						
					}
				
					std::cout << "The sum ERROR/numpoint value is: " << (double)sumerror/numResidual << endl;//观察变化趋势
					if ( iterCount == 0 || iterCount == numicp-1 )//只保存numICP-1次匹配后的特征点以及对应点
					{
						// outfeatpair.close();
						
						//实际只进行numICP-1次ICP优化 这次进行到这里只是为了查看numICP-1次ICP优化后的状态
						if(iterCount == numicp-1)
						{
							//记录overlap ratio 百分比(分母也可以是9*numselect=numscansampled) numProcessed
							// overlap_ratio = (float)numResidual / (9*numselect);
							overlap_ratio = (float)numResidual / numscansampled;//采样的点数仍固定，但总数为遍历过的总点数
							outoverlap << numResidual << " " << numscansampled << " " << numvisited <<
									" " << overlap_ratio << " " << (float)numResidual/numvisited <<endl;
							outoverlap.close();
							outangle.close();
							continue;
						}
						
					}
					printf("%d feature points are added to ResidualBlock @ %d th Iteration solver \n", numResidual, iterCount);
					
					A.conservativeResize(numResidual, NoChange);//不会改变元素！
					b.conservativeResize(numResidual, NoChange);
					loss.conservativeResize(numResidual, NoChange);
					
					if( ( int(A.rows()) != numResidual ) || ( int(A.cols()) != 6 ) || ( int(b.rows()) != numResidual ) || ( int(b.cols()) != 1 ) )
					{
						std::cout<<"Shape ERROR !"<<endl;
					}
					
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
					// std::cout<< summary.BriefReport() <<endl;
					*/
					
					//SVD求解线性最小二乘问题 https://eigen.tuxfamily.org/dox/classEigen_1_1BDCSVD.html warning
					x = A.bdcSvd(ComputeThinU | ComputeThinV).solve(b);//得到了(roll,pitch,yaw,x,y,z)
					// loss = A * x - b;
					// double loss_norm = loss.norm();
					// std::cout << "The least-squares solution is:" << x.transpose() << endl;
					// std::cout << "|A*x-b|^2 is: " << loss_norm << endl;//观察
					
					
					
					//转化为四元数和位移
					double rollrad = x(0, 0);//x(0, 0)
					double pitchrad = x(1, 0);//x(1, 0)
					double yawrad = x(2, 0);//x(2, 0)
					double t_x = x(3, 0);//x(3, 0)
					double t_y = x(4, 0);//x(4, 0)
					double t_z = x(5, 0);//x(5, 0)
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
				// std::cerr<<"the final result of frame"<< frameCount <<": q_w_curr= "<< q_w_curr.coeffs().transpose() <<" t_w_curr= "<< t_w_curr.transpose() <<"\n"<<endl;
			}
			else//点太少 一般是第0帧
			{
				ROS_WARN("Current map model points num are not enough, skip Optimization ! \n");
				// std::cerr<<"the final result of frame"<< frameCount <<": q_w_curr= "<< q_w_curr.coeffs().transpose() <<" t_w_curr= "<< t_w_curr.transpose() <<"\n"<<endl;
			}
			//逆变换
			for (int i = 0; i < int(CloudSampledFeat.size()); i++)
			{
				for (size_t j = 0; j < CloudSampledFeat[i].points.size(); j++)
				{
					pointWN = CloudSampledFeat[i].points[j];
					pointAssociateTobeMapped(&pointWN, &CloudSampledFeat[i].points[j]);	
				}
			}
			pcl::PointCloud<pointxyzinormal>::Ptr RegisteredCloud(new pcl::PointCloud<pointxyzinormal>());
			//前3帧使用IMLS/gt的结果 测试
			if(frameCount < 3 )
			{
				q_w_curr = q_wodom_curr;
				t_w_curr = t_wodom_curr;
				/*
				//把当前优化过的scan所有点变换坐标系到世界坐标系下
				for (int i = 0; i < numProcessed; i++)
				{	
					pointAssociateToMap(&CloudProcessed->points[i], &CloudProcessed->points[i]);
					CloudProcessed->points[i].intensity = frameCount;//加入地图时 添加帧数信息 方便定位哪一帧

					//debug: 注册时 用真值/imls 这样就得到gt地图 用来匹配  目的为了排除其他因素 看每次匹配的结果
					// pointAssociateToMap_gt(&CloudProcessed->points[i], &CloudProcessed->points[i], frameCount);
				}
				//test已经注册的当前帧点 在世界坐标系下重新计算法线
				// pcl::PointCloud<pointxyzinormal>::Ptr RegisteredCloud(new pcl::PointCloud<pointxyzinormal>());
				std::cout<<"RE-Computing normal for Scan-Registered "<<frameCount<<" ..."<<endl;
				RegisteredCloud = compute_normal(CloudProcessed); //新的带法线的注册后的当前帧
				*/
			}
			/*else
			{
				for (int i = 0; i < numProcessed; i++)
				{	
					ScanWN->points[i].intensity = frameCount;//加入地图时 添加帧数信息 方便定位哪一帧
					RegisteredCloud->push_back(ScanWN->points[i]);
					//debug: 注册时 用真值/imls 这样就得到gt地图 用来匹配  目的为了排除其他因素 看每次匹配的结果
					// pointAssociateToMap_gt(&CloudProcessed->points[i], &CloudProcessed->points[i], frameCount);
				}
			}
			*/
			//把当前优化过的scan所有点变换坐标系到世界坐标系下
			for (int i = 0; i < numProcessed; i++)
			{	
				pointAssociateToMap(&CloudProcessed->points[i], &CloudProcessed->points[i]);
				CloudProcessed->points[i].intensity = frameCount;//加入地图时 添加帧数信息 方便定位哪一帧

				//debug: 注册时 用真值/imls 这样就得到gt地图 用来匹配  目的为了排除其他因素 看每次匹配的结果
				// pointAssociateToMap_gt(&CloudProcessed->points[i], &CloudProcessed->points[i], frameCount);
			}
			//test已经注册的当前帧点 在世界坐标系下重新计算法线
			// pcl::PointCloud<pointxyzinormal>::Ptr RegisteredCloud(new pcl::PointCloud<pointxyzinormal>());
			std::cout<<"RE-Computing normal for Scan-Registered "<<frameCount<<" ..."<<endl;
			RegisteredCloud = compute_normal(CloudProcessed); //新的带法线的注册后的当前帧
			
			std::cerr<<"the final result of frame"<< frameCount <<": q_w_curr= "<< q_w_curr.coeffs().transpose() 
			<<" t_w_curr= "<< t_w_curr.transpose() <<"\n"<<endl;
			//迭代优化结束 更新相关的转移矩阵 选择是否更新
			// transformUpdate(); //更新了odo world 相对于 map world的变换
			// std::cout<<"the 'odo world to map world' pose of frame"<<frameCount<<": q= "<<q_wmap_wodom.coeffs().transpose()<<" t= "<<t_wmap_wodom.transpose()<<"\n"<<endl;
			
			
			//间隔地加入现有地图 间隔的帧数为(kframespace-1)
			//if (frameCount % kframespace == 0)
			// if ( overlap_ratio < min_overlap || frameCount==0 )//overlap小于阈值 才加入地图！
			if(1)
			{
				//再有间隔的情况下 帧序数
				// int mapCount = frameCount / kframespace;
				TicToc t_add;
				
				//将当前帧的点加入到modelpoint 中 相应位置
				if (frameCount<mapsize)//说明当前model point 还没存满 直接添加
				{
					for (int i = 0; i < numProcessed; i++)//把当前优化过的scan所有点注册到地图数组指针中
					{
						ModelPointCloud[frameCount].push_back(RegisteredCloud->points[i]);
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
							// std::cout<<"num of ModelPointCloud["<<j<<"] : "<< int(ModelPointCloud[j].points.size())<<endl;
						}
						
						// ModelPointCloud[j] = ModelPointCloud[j+1];
					}
					// ModelPointCloud[mapsize-1].reset(new pcl::PointCloud<PointType>());
					ModelPointCloud[mapsize-1].clear();//->
					if(int(ModelPointCloud[mapsize-1].points.size()) != 0)
					{
						std::cout<<"ERROR when clear modelpointcloud[mapsize-1]! "<<endl;
						// std::cout<<"num of ModelPointCloud["<<j<<"] : "<< int(ModelPointCloud[j].points.size) <<"\n"<<endl;
					}
					//把当前帧的点注册后加入到数组最后一个位置
					for (int i = 0; i < numProcessed; i++)
					{
						ModelPointCloud[mapsize-1].push_back(RegisteredCloud->points[i]);
					}
					if(int(ModelPointCloud[mapsize-1].points.size()) != numProcessed)
					{
						std::cout<<"ERROR when add point to modelpointcloud[99]! "<<endl;
					}
					
				}
				printf("KF-add Frame %d th points time %f ms\n", frameCount, t_add.toc());
				
				// mapCount++;//累计关键帧数，也是目前地图中的帧数
				
			}
			
			
			TicToc t_pub;
			//发布
			sensor_msgs::PointCloud2 CloudbfPreProcess;//这里是所有预处理之前的点云
			pcl::toROSMsg(*laserCloudFullRes, CloudbfPreProcess);
			CloudbfPreProcess.header.stamp = ros::Time().fromSec(timeLaserCloudFullRes);
			CloudbfPreProcess.header.frame_id = "/camera_init"; ///camera_init
			pubCloudbfPreProcess.publish(CloudbfPreProcess);
			// for (int i = 0; i < int(CloudSampled->points.size()); i++)
			// {//变换到当前坐标系
			// 	pointAssociateTobeMapped(&CloudSampled->points[i], &CloudSampled->points[i]);
			// }

			//去除动态物体前的点云
			sensor_msgs::PointCloud2 bfDOR;//
			pcl::toROSMsg(*Outlierm, bfDOR);
			bfDOR.header.stamp = ros::Time().fromSec(timeLaserCloudFullRes);
			bfDOR.header.frame_id = "/camera_init";
			pubCloudbfDOR.publish(bfDOR);

			//地面点
			sensor_msgs::PointCloud2 CloudGround;
			pcl::toROSMsg(*Ground_cloud, CloudGround);
			CloudGround.header.stamp = ros::Time().fromSec(timeLaserCloudFullRes);
			CloudGround.header.frame_id = "/camera_init";
			pubCloudGround.publish(CloudGround);

			//运动物体 
			sensor_msgs::PointCloud2 CloudDynamicObj;
			pcl::toROSMsg(*DynamicObj, CloudDynamicObj);
			CloudDynamicObj.header.stamp = ros::Time().fromSec(timeLaserCloudFullRes);
			CloudDynamicObj.header.frame_id = "/camera_init";
			pubDynamicObj.publish(CloudDynamicObj);

			sensor_msgs::PointCloud2 CloudafPreProcess;//这里是所有预处理之后的点云
			pcl::toROSMsg(*ScanWNO, CloudafPreProcess);
			CloudafPreProcess.header.stamp = ros::Time().fromSec(timeLaserCloudFullRes);
			CloudafPreProcess.header.frame_id = "/camera_init";
			pubCloudafPreProcess.publish(CloudafPreProcess);
			
			
			//for now 发布采样后的特征点
			// sensor_msgs::PointCloud2 laserCloudSampled;//
			// pcl::toROSMsg(*CloudSampled, laserCloudSampled);
			// laserCloudSampled.header.stamp = ros::Time().fromSec(timeLaserCloudFullRes);
			// laserCloudSampled.header.frame_id = "/camera_init";
			// pubCloudSampled.publish(laserCloudSampled);

			// pub each sampled feature list
			if(PUB_EACH_List)
			{
				for(int i = 0; i< 9; i++)
				{	
					//输出的是去除外点后的特征点
					// pcl::PointCloud<pointxyzinormal> ListCloud;
					// for (int j = 0; j < int(CloudSampledFeat[i].points.size()); j++)
					// {
					// 	pointWN = CloudSampledFeat[i].points[j];
					// 	if (pointWN.intensity < fdist_thr)//每个列表阈值不同fdistThs[i]  || i == 1
					// 	{
					// 		ListCloud.push_back(pointWN);
					// 	}
						
					// }
					
					sensor_msgs::PointCloud2 ListMsg;
					pcl::toROSMsg(CloudSampledFeat[i], ListMsg);//CloudSampledFeat[i] ListCloud
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

			//将点云中全部点转移到世界坐标系下  
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

	pubCloudbfPreProcess = nh.advertise<sensor_msgs::PointCloud2>("/cloud_before_preprocess", 100);//发布所有预处理前的点
	pubCloudbfDOR = nh.advertise<sensor_msgs::PointCloud2>("/cloud_before_dynamicorm", 100);// 动态物体去除前的点{已经下采样}
	pubCloudGround = nh.advertise<sensor_msgs::PointCloud2>("/cloud_ground", 100);// 分离出的地面点  
	pubDynamicObj = nh.advertise<sensor_msgs::PointCloud2>("/Dynamic_Obj", 100);//去除的运动物体
	pubCloudafPreProcess = nh.advertise<sensor_msgs::PointCloud2>("/cloud_after_preprocess", 100); //所有与处理之后,动态物体去除了
 
	// pubCloudSampled = nh.advertise<sensor_msgs::PointCloud2>("/laser_cloud_sampled", 100);//发布当前采样后的点

	// pubLaserCloudSurround = nh.advertise<sensor_msgs::PointCloud2>("/laser_cloud_surround", 100);//周围的地图点 5帧

	// pubLaserCloudMap = nh.advertise<sensor_msgs::PointCloud2>("/laser_cloud_map", 100);//更多的地图点

	// pubLaserCloudFullRes = nh.advertise<sensor_msgs::PointCloud2>("/velodyne_cloud_registered", 100);//当前帧（已注册）的所有点

	pubOdomAftMapped = nh.advertise<nav_msgs::Odometry>("/aft_mapped_to_init", 100);//优化后的位姿？

	//pubOdomAftMappedHighFrec = nh.advertise<nav_msgs::Odometry>("/aft_mapped_to_init_high_frec", 100);//接收odo的位姿 高频输出 不是最后的优化结果

	pubLaserAfterMappedPath = nh.advertise<nav_msgs::Path>("/aft_mapped_path", 100);

	if(PUB_EACH_List)//默认false
    {
        for(int i = 0; i < 9; i++) //
        {
            ros::Publisher tmp = nh.advertise<sensor_msgs::PointCloud2>("/feature_listid_" + std::to_string(i), 100);
            pubEachFeatList.push_back(tmp);
        }
    }
	//载入IMLS/gt给出 的轨迹 测试 对应的点的匹配情况  imls:/home/dlr/imlslam/src/A-LOAM/IMLStraj/04_lidar.txt
	//也可以用来维护一个gt 地图 这样我可以每一帧都和地图匹配
	bool flagr1 = loadPoses("/home/dlr/Downloads/imlsresult/imls-04-lidar.txt");//gt:"/home/dlr/Datasets/kitti-benchmark/dataset/poses/04_lidar.txt"
	bool flagr2 = loadPoses_gt("/home/dlr/kitti/dataset/poses/04_lidar.txt");
	if(flagr1 == false || flagr2 == false)
	{
		cerr<<"failed to load traj !"<<endl;
		return 1;
	}
	else
	{
		cout<<"pose size: "<<pose_gt.size()<<endl;
	}
	//mapping过程 单开一个线程
	std::thread mapping_process{imls};

	ros::spin();

	return 0;
}
