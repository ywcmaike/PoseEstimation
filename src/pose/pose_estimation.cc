#include <iostream>
#include <fstream>
#include <sstream>


#include <Eigen/Core>

#include "base/camera.h"
#include "base/pose.h"
#include "estimators/pose.h"
#include "util/math.h"
#include "util/misc.h"
#include "util/logging.h"

namespace PE{
size_t RansacPnP(const std::vector<Eigen::Vector2d> &points2D, const std::vector<Eigen::Vector3d> &points3D,
                 PE::Camera &camera, Eigen::Vector4d &qvec, Eigen::Vector3d &tvec,
                 std::vector<char> &mask, const  AbsolutePoseEstimationOptions &options) {
    size_t num_inlier = 0;


    PE::EstimateAbsolutePoseRANSAC(options, points2D, points3D, &qvec, &tvec, &camera, &num_inlier, &mask);

    return num_inlier;
}

size_t LORansacPnP(const std::vector<Eigen::Vector2d> &points2D, const std::vector<Eigen::Vector3d> &points3D,
                   PE::Camera &camera, Eigen::Vector4d &qvec, Eigen::Vector3d &tvec,
                   std::vector<char> &mask, const  AbsolutePoseEstimationOptions &options) {
    size_t num_inlier = 0;


    PE::EstimateAbsolutePoseLORANSAC(options, points2D, points3D, &qvec, &tvec, &camera, &num_inlier, &mask);

    return num_inlier;
}

size_t WeightedRansacPnP(const std::vector<Eigen::Vector2d> &points2D, const std::vector<Eigen::Vector3d> &points3D,
                         const std::vector<double> &priors,
                         PE::Camera &camera, Eigen::Vector4d &qvec, Eigen::Vector3d &tvec,
                         std::vector<char> &mask, const  AbsolutePoseEstimationOptions &options) {
    size_t num_inlier = 0;


    PE::EstimateAbsolutePoseWeightedRANSAC(options, points2D, points3D, priors, &qvec, &tvec, &camera, &num_inlier, &mask);

    return num_inlier;
}

size_t WeightedLORansacPnP(const std::vector<Eigen::Vector2d> &points2D, const std::vector<Eigen::Vector3d> &points3D,
                           const std::vector<double> &priors,
                           PE::Camera &camera, Eigen::Vector4d &qvec, Eigen::Vector3d &tvec,
                           std::vector<char> &mask, const  AbsolutePoseEstimationOptions &options) {
    size_t num_inlier = 0;

    
    PE::EstimateAbsolutePoseWeightedLORANSAC(options, points2D, points3D, priors, &qvec, &tvec, &camera, &num_inlier, &mask);

    return num_inlier;
}

size_t ProsacPnP(const std::vector<Eigen::Vector2d> &points2D, const std::vector<Eigen::Vector3d> &points3D,
                 const std::vector<double> &priors,
                 PE::Camera &camera, Eigen::Vector4d &qvec, Eigen::Vector3d &tvec,
                 std::vector<char> &mask, const  AbsolutePoseEstimationOptions &options) {
    size_t num_inlier = 0;

    std::vector<std::pair<size_t, double>> indices_priors(priors.size());
    for (size_t i = 0; i != priors.size(); ++i) {
        indices_priors[i].first = i;
        indices_priors[i].second = priors[i];
    }
    std::sort(indices_priors.begin(), indices_priors.end(), [](const std::pair<size_t, double> &a, const std::pair<size_t, double> &b) {
        return a.second > b.second;
    });
    std::vector<size_t> indices(priors.size());
    for (size_t i = 0; i != priors.size(); ++i) {
        indices[i] = indices_priors[i].first;
    }

    std::vector<Eigen::Vector2d> points2D_copy(points2D.size());
    std::vector<Eigen::Vector3d> points3D_copy(points3D.size());
    for (size_t i = 0; i != indices.size(); ++i) {
        points2D_copy[i] = points2D[indices[i]];
        points3D_copy[i] = points3D[indices[i]];
    }

    std::vector<char> mask_copy;
    PE::EstimateAbsolutePosePROSAC(options, points2D_copy, points3D_copy, &qvec, &tvec, &camera, &num_inlier, &mask_copy);

    if (mask_copy.empty())
        return 0;

    mask.resize(mask_copy.size());
    for (size_t i = 0; i != indices.size(); ++i) {
        mask[indices[i]] = mask_copy[i];
    }

    return num_inlier;
}

size_t LOProsacPnP(const std::vector<Eigen::Vector2d> &points2D, const std::vector<Eigen::Vector3d> &points3D,
                   const std::vector<double> &priors,
                   PE::Camera &camera, Eigen::Vector4d &qvec, Eigen::Vector3d &tvec,
                   std::vector<char> &mask, const  AbsolutePoseEstimationOptions &options) {
    size_t num_inlier = 0;

    std::vector<std::pair<size_t, double>> indices_priors(priors.size());
    for (size_t i = 0; i != priors.size(); ++i) {
        indices_priors[i].first = i;
        indices_priors[i].second = priors[i];
    }
    std::sort(indices_priors.begin(), indices_priors.end(), [](const std::pair<size_t, double> &a, const std::pair<size_t, double> &b) {
        return a.second > b.second;
    });
    std::vector<size_t> indices(priors.size());
    for (size_t i = 0; i != priors.size(); ++i) {
        indices[i] = indices_priors[i].first;
    }

    std::vector<Eigen::Vector2d> points2D_copy(points2D.size());
    std::vector<Eigen::Vector3d> points3D_copy(points3D.size());
    for (size_t i = 0; i != indices.size(); ++i) {
        points2D_copy[i] = points2D[indices[i]];
        points3D_copy[i] = points3D[indices[i]];
    }

    std::vector<char> mask_copy;
    PE::EstimateAbsolutePoseLOPROSAC(options, points2D_copy, points3D_copy, &qvec, &tvec, &camera, &num_inlier, &mask_copy);

    for (size_t i = 0; i != indices.size(); ++i) {
        mask[indices[i]] = mask_copy[i];
    }

    return num_inlier;
}

size_t RefinePose(const std::vector<Eigen::Vector2d> &points2D, const std::vector<Eigen::Vector3d> &points3D,
                  PE::Camera &camera, Eigen::Vector4d &qvec, Eigen::Vector3d &tvec,
                  std::vector<char> &mask) {
    PE::AbsolutePoseRefinementOptions refine_options;
    for (size_t i = 0; i != 4; ++i) {
        RefineAbsolutePose(refine_options, mask, points2D, points3D, &qvec, &tvec, &camera);
    }
    size_t num_inlier = std::count(mask.begin(), mask.end(), 1);

    return num_inlier;
}

size_t RefinePoseWithPrior(const std::vector<Eigen::Vector2d> &points2D, const std::vector<Eigen::Vector3d> &points3D,
                           const std::vector<double> &priors,
                           PE::Camera &camera, Eigen::Vector4d &qvec, Eigen::Vector3d &tvec,
                           std::vector<char> &mask) {
                               PE::AbsolutePoseRefinementOptions refine_options;
    for (size_t i = 0; i != 1; ++i) {
        RefineAbsolutePoseWithPrior(refine_options, mask, points2D, points3D, priors, &qvec, &tvec, &camera);
    }
    size_t num_inlier = std::count(mask.begin(), mask.end(), 1);

    return num_inlier;
}
}