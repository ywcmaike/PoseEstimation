#include <vector>

#include <Eigen/Core>

#include "base/camera.h"

namespace PE{

size_t RansacPnP(const std::vector<Eigen::Vector2d> &points2D, const std::vector<Eigen::Vector3d> &points3D,
                 PE::Camera &camera, Eigen::Vector4d &qvec, Eigen::Vector3d &tvec,
                 std::vector<char> &mask, const  AbsolutePoseEstimationOptions &options);

size_t LORansacPnP(const std::vector<Eigen::Vector2d> &points2D, const std::vector<Eigen::Vector3d> &points3D,
                   PE::Camera &camera, Eigen::Vector4d &qvec, Eigen::Vector3d &tvec,
                   std::vector<char> &mask, const  AbsolutePoseEstimationOptions &options);

size_t WeightedRansacPnP(const std::vector<Eigen::Vector2d> &points2D, const std::vector<Eigen::Vector3d> &points3D,
                         const std::vector<double> &priors,
                         PE::Camera &camera, Eigen::Vector4d &qvec, Eigen::Vector3d &tvec,
                         std::vector<char> &mask, const  AbsolutePoseEstimationOptions &options);

size_t WeightedLORansacPnP(const std::vector<Eigen::Vector2d> &points2D, const std::vector<Eigen::Vector3d> &points3D,
                           const std::vector<double> &priors,
                           PE::Camera &camera, Eigen::Vector4d &qvec, Eigen::Vector3d &tvec,
                           std::vector<char> &mask, const  AbsolutePoseEstimationOptions &options);

size_t ProsacPnP(const std::vector<Eigen::Vector2d> &points2D, const std::vector<Eigen::Vector3d> &points3D,
                 const std::vector<double> &priors,
                 PE::Camera &camera, Eigen::Vector4d &qvec, Eigen::Vector3d &tvec,
                 std::vector<char> &mask, const  AbsolutePoseEstimationOptions &options);

size_t LOProsacPnP(const std::vector<Eigen::Vector2d> &points2D, const std::vector<Eigen::Vector3d> &points3D,
                   const std::vector<double> &priors,
                   PE::Camera &camera, Eigen::Vector4d &qvec, Eigen::Vector3d &tvec,
                   std::vector<char> &mask, const  AbsolutePoseEstimationOptions &options);


size_t RefinePose(const std::vector<Eigen::Vector2d> &points2D, const std::vector<Eigen::Vector3d> &points3D,
                  PE::Camera &camera, Eigen::Vector4d &qvec, Eigen::Vector3d &tvec,
                  std::vector<char> &mask);

size_t RefinePoseWithPrior(const std::vector<Eigen::Vector2d> &points2D, const std::vector<Eigen::Vector3d> &points3D,
                           const std::vector<double> &priors,
                           PE::Camera &camera, Eigen::Vector4d &qvec, Eigen::Vector3d &tvec,
                           std::vector<char> &mask);
}