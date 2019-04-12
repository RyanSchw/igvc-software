#include <igvc_utils/NodeUtils.hpp>

#include <pcl/point_types.h>
#include <pcl_ros/point_cloud.h>
#include <pcl_ros/transforms.h>
#include "ros_trajectory_controller.h"
#include "trajectory_utils.h"

ROSTrajectoryController::ROSTrajectoryController()
{
  ros::NodeHandle nh;
  ros::NodeHandle pNh("~");

  std::string topic_trajectory;

  igvc::getParam(pNh, "topics/trajectory", topic_trajectory);

  trajectory_pub_ = nh.advertise<pcl::PointCloud<pcl::PointXYZ>>(topic_trajectory, 1);

  ros::Rate rate(30);
//  while (ros::ok())
//  {
    testSignedDistanceField();
//    rate.sleep();
//  }
}

void ROSTrajectoryController::testSignedDistanceField()
{
  SignedDistanceFieldOptions options = {
    .grid_rows = 10, .grid_cols = 10, .cv_type = CV_8U, .grid_x = 5, .grid_y = 5, .max_iterations = 50000
  };

  nav_msgs::Path path;
  geometry_msgs::PoseStamped pose;
  pose.pose.position.x = 0;
  pose.pose.position.y = 0;
  path.poses.emplace_back(pose);
  pose.pose.position.x = 5;
  pose.pose.position.y = 5;
  path.poses.emplace_back(pose);

  cv::Mat traversal_costs(options.grid_rows, options.grid_cols, options.cv_type, cvScalar(1u));
  cv::Mat distanceField = trajectory_utils::getSignedDistanceField(path, 0, 1, options, traversal_costs, 1.0);

  publishAsPCL(trajectory_pub_, distanceField, 1.0, "/odom", pcl_conversions::toPCL(ros::Time::now()));
}

void ROSTrajectoryController::publishAsPCL(const ros::Publisher &pub, const cv::Mat &mat, double resolution, const std::string &frame_id,
                                           uint64_t stamp)
{
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr pointcloud = boost::make_shared<pcl::PointCloud<pcl::PointXYZRGB>>();
  for (int i = 0; i < mat.cols; i++)
  {
    for (int j = 0; j < mat.rows; j++)
    {
      pcl::PointXYZRGB p{};
      p.x = static_cast<float>((i * resolution) - (mat.cols / 2.0));
      p.y = static_cast<float>((j * resolution) - (mat.rows / 2.0));
      uchar distance = mat.at<uchar>(i, j);
      p.r = (128*cos(static_cast<float>(distance)/3.0)) + 128;
      p.g = (128*sin(static_cast<float>(distance)/3.0)) + 128;
      p.b = (128*cos(static_cast<float>(distance)/3.0)) + 128;
      pointcloud->points.push_back(p);
    }
  }
  pointcloud->header.frame_id = frame_id;
  pointcloud->header.stamp = stamp;
  pub.publish(pointcloud);
}