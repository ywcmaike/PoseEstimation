//
// Created by yuhailin.
//

#ifndef VLOCTIO_ESTIMATORS_POSE_H
#define VLOCTIO_ESTIMATORS_POSE_H

#include <vector>

#include <Eigen/Core>

#include <ceres/ceres.h>

#include "base/camera.h"
#include "base/camera_models.h"
#include "base/pose.h"
#include "estimators/loransac.h"
#include "util/alignment.h"
#include "util/logging.h"
#include "util/threading.h"
#include "util/types.h"

namespace PE {

struct AbsolutePoseEstimationOptions {
    // Whether to estimate the focal length.
    bool estimate_focal_length = false;

    // Number of discrete samples for focal length estimation.
    size_t num_focal_length_samples = 30;

    // Minimum focal length ratio for discrete focal length sampling
    // around focal length of given camera.
    double min_focal_length_ratio = 0.2;

    // Maximum focal length ratio for discrete focal length sampling
    // around focal length of given camera.
    double max_focal_length_ratio = 5;

    // Number of threads for parallel estimation of focal length.
    int num_threads = ThreadPool::kMaxNumThreads;

    // Options used for P3P RANSAC.
    RANSACOptions ransac_options;

    void Check() const {
        CHECK_GT(num_focal_length_samples, 0);
        CHECK_GT(min_focal_length_ratio, 0);
        CHECK_GT(max_focal_length_ratio, 0);
        CHECK_LT(min_focal_length_ratio, max_focal_length_ratio);
        ransac_options.Check();
    }
};

struct AbsolutePoseRefinementOptions {
    // Convergence criterion.
    double gradient_tolerance = 1.0;

    // Maximum number of solver iterations.
    int max_num_iterations = 100;

    // Scaling factor determines at which residual robustification takes place.
    double loss_function_scale = 1.0;

    // Whether to refine the focal length parameter group.
    bool refine_focal_length = false;

    // Whether to refine the extra parameter group.
    bool refine_extra_params = false;

    // Whether to print final summary.
    bool print_summary = false;

    void Check() const {
        CHECK_GE(gradient_tolerance, 0.0);
        CHECK_GE(max_num_iterations, 0);
        CHECK_GE(loss_function_scale, 0.0);
    }
};


bool EstimateAbsolutePoseRANSAC(const AbsolutePoseEstimationOptions& options,
                               const std::vector<Eigen::Vector2d>& points2D,
                               const std::vector<Eigen::Vector3d>& points3D,
                               Eigen::Vector4d* qvec, Eigen::Vector3d* tvec,
                               Camera* camera, size_t* num_inliers,
                               std::vector<char>* inlier_mask);

bool EstimateAbsolutePoseLORANSAC(const AbsolutePoseEstimationOptions& options,
                                  const std::vector<Eigen::Vector2d>& points2D,
                                  const std::vector<Eigen::Vector3d>& points3D,
                                  Eigen::Vector4d* qvec, Eigen::Vector3d* tvec,
                                  Camera* camera, size_t* num_inliers,
                                  std::vector<char>* inlier_mask);


bool EstimateAbsolutePoseWeightedRANSAC(const AbsolutePoseEstimationOptions &options,
                                        const std::vector<Eigen::Vector2d> &points2D,
                                        const std::vector<Eigen::Vector3d> &points3D,
                                        const std::vector<double> &distribution,
                                        Eigen::Vector4d *qvec, Eigen::Vector3d *tvec,
                                        Camera *camera, size_t *num_inliers,
                                        std::vector<char> *inlier_mask);


bool EstimateAbsolutePoseWeightedLORANSAC(const AbsolutePoseEstimationOptions &options,
                                          const std::vector<Eigen::Vector2d> &points2D,
                                          const std::vector<Eigen::Vector3d> &points3D,
                                          const std::vector<double> &distribution,
                                          Eigen::Vector4d *qvec, Eigen::Vector3d *tvec,
                                          Camera *camera, size_t *num_inliers,
                                          std::vector<char> *inlier_mask);

bool EstimateAbsolutePosePROSAC(const AbsolutePoseEstimationOptions& options,
                                const std::vector<Eigen::Vector2d>& points2D,
                                const std::vector<Eigen::Vector3d>& points3D,
                                Eigen::Vector4d* qvec, Eigen::Vector3d* tvec,
                                Camera* camera, size_t* num_inliers,
                                std::vector<char>* inlier_mask);


bool EstimateAbsolutePoseLOPROSAC(const AbsolutePoseEstimationOptions& options,
                                  const std::vector<Eigen::Vector2d>& points2D,
                                  const std::vector<Eigen::Vector3d>& points3D,
                                  Eigen::Vector4d* qvec, Eigen::Vector3d* tvec,
                                  Camera* camera, size_t* num_inliers,
                                  std::vector<char>* inlier_mask);


bool RefineAbsolutePose(const AbsolutePoseRefinementOptions& options,
                        const std::vector<char>& inlier_mask,
                        const std::vector<Eigen::Vector2d>& points2D,
                        const std::vector<Eigen::Vector3d>& points3D,
                        Eigen::Vector4d* qvec, Eigen::Vector3d* tvec,
                        Camera* camera);

bool RefineAbsolutePoseWithPrior(const AbsolutePoseRefinementOptions& options,
                                const std::vector<char>& inlier_mask,
                                const std::vector<Eigen::Vector2d>& points2D,
                                const std::vector<Eigen::Vector3d>& points3D,
                                const std::vector<double>& priors,
                                Eigen::Vector4d* qvec, Eigen::Vector3d* tvec,
                                Camera* camera);
}
#endif //VLOCTIO_ESTIMATORS_POSE_H
