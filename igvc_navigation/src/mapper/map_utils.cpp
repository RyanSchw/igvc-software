#include <unordered_set>

#include <pcl/common/common.h>
#include <pcl/features/normal_3d.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/radius_outlier_removal.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>

#include <pcl/segmentation/extract_clusters.h>
#include <pcl/segmentation/sac_segmentation.h>

#include "map_utils.h"
#include "octomapper.h"

namespace MapUtils
{
inline int discretize(radians angle, double angular_resolution)
{
  double coeff = 1 / angular_resolution;
  return static_cast<int>(std::round(coeff * angle));
}

void filterPointsBehind(const pcl::PointCloud<pcl::PointXYZ>& pc, pcl::PointCloud<pcl::PointXYZ>& filtered_pc,
                        BehindFilterOptions options)
{
  static double end_angle = -M_PI + options.angle / 2;
  static double start_angle = M_PI - options.angle / 2;
  static double squared_distance = options.distance * options.distance;
  // Iterate over pointcloud, insert discretized angles into set
  for (auto i : pc)
  {
    double angle = atan2(i.y, i.x);
    if ((-M_PI <= angle && angle < end_angle) || (start_angle < angle && angle <= M_PI))
    {
      if (i.x * i.x + i.y * i.y > squared_distance)
      {
        filtered_pc.points.emplace_back(i);
      }
    }
    else
    {
      filtered_pc.points.emplace_back(i);
    }
  }
}

std::optional<GroundPlane> filterGroundPlane(const PointCloud& raw_pc, PointCloud& ground, PointCloud& nonground,
                                             GroundFilterOptions options)
{
  ground.header = raw_pc.header;
  nonground.header = raw_pc.header;

  std::optional<GroundPlane> ground_plane = ransacFilter(raw_pc, ground, nonground, options.ransac_options);
  if (ground_plane)
  {
    return ground_plane;
  }
  else
  {
    fallbackFilter(raw_pc, ground, nonground, options.fallback_options);
    return std::nullopt;
  }
}

void projectTo2D(PointCloud& projected_pc)
{
  for (auto& point : projected_pc.points)
  {
    point.z = 0;
  }
}

std::optional<GroundPlane> ransacFilter(const PointCloud& raw_pc, PointCloud& ground, PointCloud& nonground,
                                        const RANSACOptions& options)
{
  // Plane detection for ground removal
  pcl::ModelCoefficientsPtr coefficients(new pcl::ModelCoefficients);
  pcl::PointIndices::Ptr inliers(new pcl::PointIndices);

  // create the segmentation object
  pcl::SACSegmentation<pcl::PointXYZ> seg;
  seg.setOptimizeCoefficients(true);

  seg.setModelType(pcl::SACMODEL_PERPENDICULAR_PLANE);
  seg.setMethodType(pcl::SAC_RANSAC);
  seg.setMaxIterations(options.iterations);
  seg.setDistanceThreshold(options.distance_threshold);
  seg.setAxis(Eigen::Vector3f(0, 0, 1));
  seg.setEpsAngle(options.eps_angle);

  PointCloud::Ptr cloud_filtered = raw_pc.makeShared();

  // Create filtering object
  pcl::ExtractIndices<pcl::PointXYZ> extract;

  seg.setInputCloud(cloud_filtered);
  seg.segment(*inliers, *coefficients);

  if (inliers->indices.empty())
  {
    return std::nullopt;
  }
  else
  {
    ROS_DEBUG("Ground plane found: %zu/%zu inliers. Coeff: %f %f %f %f", inliers->indices.size(),
              cloud_filtered->size(), coefficients->values.at(0), coefficients->values.at(1),
              coefficients->values.at(2), coefficients->values.at(3));
    extract.setInputCloud(cloud_filtered);
    extract.setIndices(inliers);
    extract.setNegative(false);
    extract.filter(ground);

    // remove ground points from full pointcloud
    if (inliers->indices.size() != cloud_filtered->size())
    {
      extract.setNegative(true);
      PointCloud out;
      extract.filter(out);
      nonground += out;
      *cloud_filtered = out;
    }

    return GroundPlane{ coefficients->values.at(0), coefficients->values.at(1), coefficients->values.at(2),
                        coefficients->values.at(3) };
  }
}

void fallbackFilter(const PointCloud& raw_pc, PointCloud& ground, PointCloud& nonground, const FallbackOptions& options)
{
  pcl::PassThrough<pcl::PointXYZ> fallback;
  fallback.setFilterFieldName("z");
  fallback.setFilterLimits(-options.plane_distance, options.plane_distance);
  fallback.filter(ground);

  fallback.setFilterLimitsNegative(true);
  fallback.filter(nonground);
}

void blur(cv::Mat& blurred_map, double kernel_size)
{
  cv::Mat original = blurred_map.clone();
  cv::blur(blurred_map, blurred_map, cv::Size(kernel_size, kernel_size));
  cv::max(original, blurred_map, blurred_map);
}

void getEmptyPoints(const pcl::PointCloud<pcl::PointXYZ>& pc, std::vector<Ray>& empty_rays, double angular_resolution,
                    EmptyFilterOptions options)
{
  int discretized_start = discretize(options.start_angle, angular_resolution);
  int discretized_end = discretize(options.end_angle, angular_resolution);
  empty_rays.reserve(discretized_end - discretized_start + 1);

  // Iterate over pointcloud, insert discretized angles into set if within max range
  std::unordered_set<int> discretized_angles{};
  for (auto i : pc)
  {
    if (withinRange(i, options.max_range))
    {
      double angle = atan2(i.y, i.x);
      discretized_angles.emplace(MapUtils::discretize(angle, angular_resolution));
    }
  }

  // For each angle, if it's not in the set (empty), put it into a pointcloud.
  // From Robot's frame. Need to rotate angle to world frame
  for (int i = discretized_start; i < discretized_end; i++)
  {
    if (discretized_angles.find(i) == discretized_angles.end())
    {
      double angle = i * angular_resolution;
      pcl::PointXYZ ray_start{ static_cast<float>(options.ray_start_distance * cos(angle)),
                               static_cast<float>(options.ray_start_distance * sin(angle)), 0 };

      pcl::PointXYZ ray_end{ static_cast<float>(options.miss_cast_distance * cos(angle)),
                             static_cast<float>(options.miss_cast_distance * sin(angle)), 0 };
      empty_rays.emplace_back(Ray{ ray_start, ray_end });
    }
  }
}

void projectToPlane(PointCloud& projected_pc, const GroundPlane& ground_plane, const cv::Mat& image,
                    const image_geometry::PinholeCameraModel& model, const tf::Transform& camera_to_world, bool is_line)
{
  int nRows = image.rows;
  int nCols = image.cols;

  uchar match = is_line ? 255u : 0u;

  int i, j;
  const uchar* p;
  for (i = 0; i < nRows; ++i)
  {
    p = image.ptr<uchar>(i);
    for (j = 0; j < nCols; ++j)
    {
      // If it's a black pixel => free space, then project
      if (p[j] == match)
      {
        cv::Point2d pixel_point(j, i);

        cv::Point3d ray = model.projectPixelTo3dRay(pixel_point);
        tf::Point reoriented_ray{ ray.x, ray.y, ray.z };  // cv::Point3d defined with z forward, x right, y down
        tf::Point transformed_ray = camera_to_world.getBasis() * reoriented_ray;  // Transform ray to odom frame

        // If camera sees horizon, then ray of z > 0 means that it's the sky. projecting that to the plane would
        // result in it being behinds us...
        if (transformed_ray.z() > 0)
        {
          continue;
        }

        double a = ground_plane.a;
        double b = ground_plane.b;
        double c = ground_plane.c;
        double d = ground_plane.d;

        // [a b c]^T dot (P0 - (camera + ray*t)) = 0, solve for t
        double t = ((tf::Vector3{ 0, 0, d / c } - camera_to_world.getOrigin()).dot(tf::Vector3{ a, b, c })) /
                   (transformed_ray.dot(tf::Vector3{ a, b, c }));

        // projected_point = camera + ray*t
        tf::Point projected_point = camera_to_world.getOrigin() + transformed_ray * t;
        projected_pc.points.emplace_back(pcl::PointXYZ(static_cast<float>(projected_point.x()),
                                                       static_cast<float>(projected_point.y()),
                                                       static_cast<float>(projected_point.z())));
      }
    }
  }
}

sensor_msgs::CameraInfoConstPtr scaleCameraInfo(const sensor_msgs::CameraInfoConstPtr& camera_info, double width,
                                                double height)
{
  sensor_msgs::CameraInfoPtr changed_camera_info = boost::make_shared<sensor_msgs::CameraInfo>(*camera_info);
  changed_camera_info->D = camera_info->D;
  changed_camera_info->distortion_model = camera_info->distortion_model;
  changed_camera_info->R = camera_info->R;
  changed_camera_info->roi = camera_info->roi;
  changed_camera_info->binning_x = camera_info->binning_x;
  changed_camera_info->binning_y = camera_info->binning_y;

  double waf = static_cast<double>(width) / static_cast<double>(camera_info->width);
  double haf = static_cast<double>(height) / static_cast<double>(camera_info->height);

  changed_camera_info->width = static_cast<unsigned int>(width);
  changed_camera_info->height = static_cast<unsigned int>(height);

  changed_camera_info->K = { { camera_info->K[0] * waf, 0, camera_info->K[2] * waf, 0, camera_info->K[4] * haf,
                               camera_info->K[5] * haf, 0, 0, 1 } };
  changed_camera_info->P = { { camera_info->P[0] * waf, 0, camera_info->P[2] * waf, 0, 0, camera_info->P[5] * haf,
                               camera_info->P[6] * haf, 0, 0, 0, 1, 0 } };
  return changed_camera_info;
}

void denoise(const pcl::PointCloud<pcl::PointXYZ>::ConstPtr& pointcloud, pcl::PointCloud<pcl::PointXYZ>& out, double k,
             double stddev)
{
  pcl::PointCloud<pcl::PointXYZ>::Ptr after_sor(new pcl::PointCloud<pcl::PointXYZ>);
  pcl::StatisticalOutlierRemoval<pcl::PointXYZ> sor;
  sor.setInputCloud(pointcloud);
  sor.setMeanK(k);
  sor.setStddevMulThresh(stddev);
  sor.filter(*after_sor);

  pcl::VoxelGrid<pcl::PointXYZ> voxel_grid;
  voxel_grid.setInputCloud(after_sor);
  voxel_grid.setLeafSize(0.1, 0.1, 0.1);
  voxel_grid.filter(out);
}

std::vector<pcl::PointIndices> cluster(const pcl::PointCloud<pcl::PointXYZ>::ConstPtr& pointcloud,
                                       const pcl::search::KdTree<pcl::PointXYZ>::Ptr& tree, double tolerance)
{
  pcl::PointCloud<pcl::PointXYZ>::Ptr copy(new pcl::PointCloud<pcl::PointXYZ>(*pointcloud));
  projectTo2D(*copy);

  std::vector<pcl::PointIndices> cluster_indices;
  pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
  ec.setClusterTolerance(tolerance);  // 2cm
  ec.setMinClusterSize(10);
  ec.setMaxClusterSize(500);
  ec.setSearchMethod(tree);
  ec.setInputCloud(copy);
  ec.extract(cluster_indices);

  return cluster_indices;
}

void cluster(const pcl::PointCloud<pcl::PointXYZ>::ConstPtr& pointcloud, pcl::PointCloud<pcl::PointXYZI>& out,
             const pcl::search::KdTree<pcl::PointXYZ>::Ptr& tree, double tolerance)
{
  auto cluster_indices = cluster(pointcloud, tree, tolerance);

  int j = 0;
  for (const auto& cluster_indice : cluster_indices)
  {
    for (int index : cluster_indice.indices)
    {
      const auto point = pointcloud->points[index];
      pcl::PointXYZI xyzi{};
      xyzi.x = point.x;
      xyzi.y = point.y;
      xyzi.z = point.z;
      xyzi.intensity = j;
      out.points.emplace_back(xyzi);  //*
    }
    j++;
  }
}

std::array<std::vector<pcl::PointCloud<pcl::PointXYZ>>, 3>
extractBarrels(const pcl::PointCloud<pcl::PointXYZ>::ConstPtr& pointcloud, const tf::Vector3& origin, double threshold)
{
  pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
  auto cluster_indices = cluster(pointcloud, tree, 0.15);

  std::vector<pcl::PointCloud<pcl::PointXYZ>> barrels;
  std::vector<pcl::PointCloud<pcl::PointXYZ>> clusters;
  std::vector<pcl::PointCloud<pcl::PointXYZ>> rejected;

  for (const auto& cluster_index : cluster_indices)
  {
    auto dims = calculatePointcloudDims(*pointcloud, cluster_index);
    if (dims[2] < 0.25)
    {
      rejected.emplace_back(pointcloudFromIndices(pointcloud, boost::make_shared<pcl::PointIndices>(cluster_index)));
      continue;
    }

    if (cluster_index.indices.size() > 250)
    {
      rejected.emplace_back(pointcloudFromIndices(pointcloud, boost::make_shared<pcl::PointIndices>(cluster_index)));
      continue;
    }

    auto extracted =
        extractBarrel(pointcloud, boost::make_shared<pcl::PointIndices>(cluster_index), tree, origin, threshold);
    if (!extracted->indices.empty())
    {
      auto barrel_dims = calculatePointcloudDims(*pointcloud, *extracted);
      if (barrel_dims[0] > 0.7 || barrel_dims[1] > 0.7)
      {
        rejected.emplace_back(pointcloudFromIndices(pointcloud, boost::make_shared<pcl::PointIndices>(cluster_index)));
        continue;
      }
      barrels.emplace_back();
    }
    clusters.emplace_back(pointcloudFromIndices(pointcloud, boost::make_shared<pcl::PointIndices>(cluster_index)));
  }

  return { barrels, clusters, rejected };
}

Eigen::Vector4f calculatePointcloudDims(const pcl::PointCloud<pcl::PointXYZ>& pointcloud,
                                        const pcl::PointIndices& indicies)
{
  Eigen::Vector4f min_pt{};
  Eigen::Vector4f max_pt{};

  pcl::getMinMax3D(pointcloud, indicies, min_pt, max_pt);
  return max_pt - min_pt;
}

pcl::PointCloud<pcl::PointXYZI> vectorToIntensity(const std::vector<pcl::PointCloud<pcl::PointXYZ>>& pointclouds)
{
  pcl::PointCloud<pcl::PointXYZI> out{};
  int index = 0;
  for (const auto& pointcloud : pointclouds)
  {
    for (const auto& point : pointcloud.points)
    {
      pcl::PointXYZI xyzi{};
      xyzi.x = point.x;
      xyzi.y = point.y;
      xyzi.z = point.z;
      xyzi.intensity = index;
      out.points.emplace_back(xyzi);
    }
    index++;
  }

  return out;
}

pcl::PointIndices::Ptr extractBarrel(const pcl::PointCloud<pcl::PointXYZ>::ConstPtr& pointcloud,
                                     const pcl::PointIndices::Ptr& indices,
                                     const pcl::search::KdTree<pcl::PointXYZ>::Ptr& tree, const tf::Vector3& origin,
                                     double threshold)
{
  pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> ne;
  pcl::SACSegmentation<pcl::PointXYZ> seg;
  pcl::ExtractIndices<pcl::PointXYZ> extract;
  pcl::ExtractIndices<pcl::Normal> extract_normals;

  pcl::PointCloud<pcl::Normal>::Ptr cloud_normals(new pcl::PointCloud<pcl::Normal>);

  pcl::PointCloud<pcl::PointXYZ>::Ptr flattened =
      boost::make_shared<pcl::PointCloud<pcl::PointXYZ>>(pointcloudFromIndices(pointcloud, indices));
  projectTo2D(*flattened);
  //
  //  ne.setSearchMethod(tree);
  //  ne.setInputCloud(flattened);
  //  ne.setRadiusSearch(0.1);
  //  ne.setViewPoint(origin.x(), origin.y(), origin.z());
  //  ne.compute(*cloud_normals);

  // PROJECT TO 2D, get indices, then yay ^.^
  seg.setOptimizeCoefficients(true);
  seg.setModelType(pcl::SACMODEL_CIRCLE2D);
  seg.setMethodType(pcl::SAC_RANSAC);
  seg.setMaxIterations(10000);
  seg.setDistanceThreshold(threshold);
  seg.setRadiusLimits(0.2, 0.3);
  seg.setAxis({ 0, 0, 1 });
  seg.setInputCloud(flattened);
  seg.setIndices(indices);

  pcl::PointIndices::Ptr inliers_cylinder(new pcl::PointIndices);
  pcl::ModelCoefficients::Ptr coefficients_cylinder(new pcl::ModelCoefficients);
  seg.segment(*inliers_cylinder, *coefficients_cylinder);

  return inliers_cylinder;
}

pcl::PointCloud<pcl::PointXYZ> pointcloudFromIndices(const pcl::PointCloud<pcl::PointXYZ>::ConstPtr& pointcloud,
                                                     const pcl::PointIndices::ConstPtr& indices)
{
  pcl::PointCloud<pcl::PointXYZ> out{};

  pcl::ExtractIndices<pcl::PointXYZ> extract;
  extract.setInputCloud(pointcloud);
  extract.setIndices(indices);
  extract.setNegative(false);
  extract.filter(out);

  return out;
}

void extractBarrels(const pcl::PointCloud<pcl::PointXYZ>::ConstPtr& pointcloud, pcl::PointCloud<pcl::PointXYZ>& out,
                    const tf::Vector3& origin)
{
  pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> ne;
  pcl::SACSegmentationFromNormals<pcl::PointXYZ, pcl::Normal> seg;
  pcl::ExtractIndices<pcl::PointXYZ> extract;
  pcl::ExtractIndices<pcl::Normal> extract_normals;
  pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>());

  pcl::PointCloud<pcl::Normal>::Ptr cloud_normals(new pcl::PointCloud<pcl::Normal>);

  ne.setSearchMethod(tree);
  ne.setInputCloud(pointcloud);
  ne.setRadiusSearch(0.1);
  ne.setViewPoint(origin.x(), origin.y(), origin.z());
  ne.compute(*cloud_normals);

  seg.setOptimizeCoefficients(true);
  seg.setModelType(pcl::SACMODEL_CYLINDER);
  seg.setMethodType(pcl::SAC_RANSAC);
  seg.setNormalDistanceWeight(0.1);
  seg.setMaxIterations(10000);
  seg.setDistanceThreshold(0.01);
  seg.setRadiusLimits(0.2, 0.3);
  seg.setAxis({ 0, 0, 1 });
  seg.setInputCloud(pointcloud);
  seg.setInputNormals(cloud_normals);

  pcl::PointIndices::Ptr inliers_cylinder(new pcl::PointIndices);
  pcl::ModelCoefficients::Ptr coefficients_cylinder(new pcl::ModelCoefficients);
  seg.segment(*inliers_cylinder, *coefficients_cylinder);

  ROS_ERROR_STREAM("Radius: " << coefficients_cylinder->values[6]);
  ROS_ERROR_STREAM("Coeffs: " << *coefficients_cylinder << "\n");

  extract.setInputCloud(pointcloud);
  extract.setIndices(inliers_cylinder);
  extract.setNegative(false);
  extract.filter(out);
}
}  // namespace MapUtils
