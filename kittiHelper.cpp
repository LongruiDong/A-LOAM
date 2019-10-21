// Author:   Tong Qin               qintonguav@gmail.com
// 	         Shaozu Cao 		    saozu.cao@connect.ust.hk

#include <iostream>
#include <fstream>
using namespace std;
#include <iterator>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include <image_transport/image_transport.h>
#include <opencv2/highgui/highgui.hpp>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <ros/ros.h>
#include <rosbag/bag.h>
#include <geometry_msgs/PoseStamped.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <eigen3/Eigen/Dense>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <sensor_msgs/PointCloud2.h>

// 读取kitti lidar .bin文件
std::vector<float> read_lidar_data(const std::string lidar_data_path)
{
    std::ifstream lidar_data_file(lidar_data_path, std::ifstream::in | std::ifstream::binary);
    lidar_data_file.seekg(0, std::ios::end);
    const size_t num_elements = lidar_data_file.tellg() / sizeof(float);
    lidar_data_file.seekg(0, std::ios::beg);

    std::vector<float> lidar_data_buffer(num_elements);
    lidar_data_file.read(reinterpret_cast<char*>(&lidar_data_buffer[0]), num_elements*sizeof(float));
    return lidar_data_buffer;
}

int main(int argc, char** argv)
{
    ros::init(argc, argv, "kitti_helper");
    ros::NodeHandle n("~");
    std::string dataset_folder, sequence_number, output_bag_file;
    // /home/dlr/kitti/dataset/
    n.getParam("dataset_folder", dataset_folder);
    n.getParam("sequence_number", sequence_number);
    std::cout << "Reading sequence " << sequence_number << " from " << dataset_folder << '\n';
    bool to_bag;
    n.getParam("to_bag", to_bag);
    if (to_bag)
        n.getParam("output_bag_file", output_bag_file);
    int publish_delay; // 延迟？ 什么意义 
    n.getParam("publish_delay", publish_delay);
    publish_delay = publish_delay <= 0 ? 1 : publish_delay; // 参数值必须为正
    // 输入的激光点云节点 将在"/velodyne_points" topic上发布点云
    ros::Publisher pub_laser_cloud = n.advertise<sensor_msgs::PointCloud2>("/velodyne_points", 2);

    image_transport::ImageTransport it(n);
    // 输入的双目图像节点
    image_transport::Publisher pub_image_left = it.advertise("/image_left", 2);
    image_transport::Publisher pub_image_right = it.advertise("/image_right", 2);

    ros::Publisher pubOdomGT = n.advertise<nav_msgs::Odometry> ("/odometry_gt", 5);
    nav_msgs::Odometry odomGT;
    odomGT.header.frame_id = "/camera_init"; // 是什么
    odomGT.child_frame_id = "/ground_truth";

    ros::Publisher pubPathGT = n.advertise<nav_msgs::Path> ("/path_gt", 5); // 和odometry_gt区别？
    nav_msgs::Path pathGT;
    pathGT.header.frame_id = "/camera_init";
    
    std::string timestamp_path = "sequences/" + sequence_number + "/times.txt"; // 图像的时间戳
    std::ifstream timestamp_file(dataset_folder + timestamp_path, std::ifstream::in);

    std::string ground_truth_path = "poses/" + sequence_number + ".txt"; // 真实轨迹
    std::ifstream ground_truth_file(dataset_folder + ground_truth_path, std::ifstream::in);

    rosbag::Bag bag_out;
    if (to_bag)
        bag_out.open(output_bag_file, rosbag::bagmode::Write);
    
    Eigen::Matrix3d R_transform;
    R_transform << 0, 0, 1, -1, 0, 0, 0, -1, 0; // 坐标系的一个旋转 ,为何要对gt 坐标系作此变换
    Eigen::Quaterniond q_transform(R_transform); //对应的四元数
    cout << "q_transform: " << q_transform.coeffs() <<endl; //输出该旋转四元数: x,y,z,w 0.5 -0.5 0.5 -0.5

    std::string line;
    std::size_t line_num = 0;

    ros::Rate r(10.0 / publish_delay); //delay 影响 ros rate 会影响什么？
    while (std::getline(timestamp_file, line) && ros::ok()) //逐个图像读取
    {
        float timestamp = stof(line); // 读入时间戳
        std::stringstream left_image_path, right_image_path; // 这里读入图像只是为了显示窗口吧
        left_image_path << dataset_folder << "sequences/" + sequence_number + "/image_0/" << std::setfill('0') << std::setw(6) << line_num << ".png"; // 图像名是行数
        cv::Mat left_image = cv::imread(left_image_path.str(), CV_LOAD_IMAGE_GRAYSCALE);
        right_image_path << dataset_folder << "sequences/" + sequence_number + "/image_1/" << std::setfill('0') << std::setw(6) << line_num << ".png";
        cv::Mat right_image = cv::imread(left_image_path.str(), CV_LOAD_IMAGE_GRAYSCALE);

        std::getline(ground_truth_file, line);
        std::stringstream pose_stream(line);
        std::string s;
        Eigen::Matrix<double, 3, 4> gt_pose; // 轨迹格式是12个值
        for (std::size_t i = 0; i < 3; ++i)
        {
            for (std::size_t j = 0; j < 4; ++j)
            {
                std::getline(pose_stream, s, ' ');
                gt_pose(i, j) = stof(s);
            }
        }
        // 得到变换后的该时刻的旋转和平移 为何对gt变换？
        Eigen::Quaterniond q_w_i(gt_pose.topLeftCorner<3, 3>());
        Eigen::Quaterniond q = q_transform * q_w_i;
        q.normalize();
        Eigen::Vector3d t = q_transform * gt_pose.topRightCorner<3, 1>();

        odomGT.header.stamp = ros::Time().fromSec(timestamp);
        odomGT.pose.pose.orientation.x = q.x();
        odomGT.pose.pose.orientation.y = q.y();
        odomGT.pose.pose.orientation.z = q.z();
        odomGT.pose.pose.orientation.w = q.w();
        odomGT.pose.pose.position.x = t(0);
        odomGT.pose.pose.position.y = t(1);
        odomGT.pose.pose.position.z = t(2);
        pubOdomGT.publish(odomGT); // publish变换后的真实轨迹

        geometry_msgs::PoseStamped poseGT;
        poseGT.header = odomGT.header;
        poseGT.pose = odomGT.pose.pose;
        pathGT.header.stamp = odomGT.header.stamp;
        pathGT.poses.push_back(poseGT);
        pubPathGT.publish(pathGT); // 和odomGT的区别？

        // read lidar point cloud
        std::stringstream lidar_data_path;
        lidar_data_path << dataset_folder << "velodyne/sequences/" + sequence_number + "/velodyne/" 
                        << std::setfill('0') << std::setw(6) << line_num << ".bin"; // 当前时刻的激光点云
        std::vector<float> lidar_data = read_lidar_data(lidar_data_path.str());
        std::cout << "totally " << lidar_data.size() / 4.0 << " points in this lidar frame \n"; // 一个点4个值

        std::vector<Eigen::Vector3d> lidar_points; // 3
        std::vector<float> lidar_intensities; // 1 强度？
        pcl::PointCloud<pcl::PointXYZI> laser_cloud; // 点云数据结构
        for (std::size_t i = 0; i < lidar_data.size(); i += 4)
        {
            lidar_points.emplace_back(lidar_data[i], lidar_data[i+1], lidar_data[i+2]);
            lidar_intensities.push_back(lidar_data[i+3]);

            pcl::PointXYZI point;
            point.x = lidar_data[i];
            point.y = lidar_data[i + 1];
            point.z = lidar_data[i + 2];
            point.intensity = lidar_data[i + 3];
            laser_cloud.push_back(point);
        }
        // 信息？
        sensor_msgs::PointCloud2 laser_cloud_msg;
        pcl::toROSMsg(laser_cloud, laser_cloud_msg); 
        laser_cloud_msg.header.stamp = ros::Time().fromSec(timestamp);
        laser_cloud_msg.header.frame_id = "/camera_init";
        pub_laser_cloud.publish(laser_cloud_msg);//发布到 tpic : "/velodyne_points"

        sensor_msgs::ImagePtr image_left_msg = cv_bridge::CvImage(laser_cloud_msg.header, "mono8", left_image).toImageMsg();
        sensor_msgs::ImagePtr image_right_msg = cv_bridge::CvImage(laser_cloud_msg.header, "mono8", right_image).toImageMsg();
        pub_image_left.publish(image_left_msg);
        pub_image_right.publish(image_right_msg);

        if (to_bag) // 把图像、点云、真实位姿输出为bag
        { // 5个节点
            bag_out.write("/image_left", ros::Time::now(), image_left_msg);
            bag_out.write("/image_right", ros::Time::now(), image_right_msg);
            bag_out.write("/velodyne_points", ros::Time::now(), laser_cloud_msg);
            bag_out.write("/path_gt", ros::Time::now(), pathGT);
            bag_out.write("/odometry_gt", ros::Time::now(), odomGT);
        }

        line_num ++;
        r.sleep();
    }
    bag_out.close();
    std::cout << "Done \n";


    return 0;
}