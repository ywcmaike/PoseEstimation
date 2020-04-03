//
// Created by yuhailin.
//

#include "estimators/pose.h"

#include <vector>
#include <iostream>
#include <iomanip>

#include "base/camera_models.h"
#include "base/cost_functions.h"
#include "base/pose.h"
#include "estimators/absolute_pose.h"
#include "optim/bundle_adjustment.h"
#include "util/matrix.h"
#include "util/misc.h"
#include "util/threading.h"

namespace PE {
namespace {

// ransac
typedef RANSAC<P3PEstimator> AbsolutePoseRANSAC;
typedef LORANSAC<P3PEstimator, EPNPEstimator> AbsolutePoseLORANSAC;

// weighted ransac
typedef RANSAC<P3PEstimator, InlierSupportMeasurer, WeigthedRandomSampler> AbsolutePoseWeightedRANSAC;
typedef LORANSAC<P3PEstimator, EPNPEstimator, InlierSupportMeasurer, WeigthedRandomSampler> AbsolutePoseWeightedLORANSAC;

// prosac
typedef RANSAC<P3PEstimator, InlierSupportMeasurer, ProgressiveSampler> AbsolutePosePROSAC;
typedef LORANSAC<P3PEstimator, EPNPEstimator, InlierSupportMeasurer, ProgressiveSampler> AbsolutePoseLOPROSAC;

void EstimateAbsolutePoseRANSACKernel(const Camera& camera,
                                      const double focal_length_factor,
                                      const std::vector<Eigen::Vector2d>& points2D,
                                      const std::vector<Eigen::Vector3d>& points3D,
                                      const RANSACOptions& options,
                                      AbsolutePoseRANSAC::Report* report) {
    // Scale the focal length by the given factor.
    Camera scaled_camera = camera;
    const std::vector<size_t>& focal_length_idxs = camera.FocalLengthIdxs();
    for (const size_t idx : focal_length_idxs) {
        scaled_camera.Params(idx) *= focal_length_factor;
    }

    // Normalize image coordinates with current camera hypothesis.
    std::vector<Eigen::Vector2d> points2D_N(points2D.size());
    for (size_t i = 0; i < points2D.size(); ++i) {
        points2D_N[i] = scaled_camera.ImageToWorld(points2D[i]);
    }

    // Estimate pose for given focal length.
    auto custom_options = options;
    custom_options.max_error =
            scaled_camera.ImageToWorldThreshold(options.max_error);
    AbsolutePoseRANSAC ransac(custom_options);
    *report = ransac.Estimate(points2D_N, points3D);
}

void EstimateAbsolutePoseLORANSACKernel(const Camera& camera,
                                        const double focal_length_factor,
                                        const std::vector<Eigen::Vector2d>& points2D,
                                        const std::vector<Eigen::Vector3d>& points3D,
                                        const RANSACOptions& options,
                                        AbsolutePoseLORANSAC::Report* report) {
    // Scale the focal length by the given factor.
    Camera scaled_camera = camera;
    const std::vector<size_t>& focal_length_idxs = camera.FocalLengthIdxs();
    for (const size_t idx : focal_length_idxs) {
        scaled_camera.Params(idx) *= focal_length_factor;
    }

    // Normalize image coordinates with current camera hypothesis.
    std::vector<Eigen::Vector2d> points2D_N(points2D.size());
    for (size_t i = 0; i < points2D.size(); ++i) {
        points2D_N[i] = scaled_camera.ImageToWorld(points2D[i]);
    }

    // Estimate pose for given focal length.
    auto custom_options = options;
    custom_options.max_error =
            scaled_camera.ImageToWorldThreshold(options.max_error);
    AbsolutePoseLORANSAC ransac(custom_options);
    *report = ransac.Estimate(points2D_N, points3D);
}

void EstimateAbsolutePoseWeightedRANSACKernel(const Camera& camera,
                                              const double focal_length_factor,
                                              const std::vector<Eigen::Vector2d>& points2D,
                                              const std::vector<Eigen::Vector3d>& points3D,
                                              const std::vector<double>& distribution,
                                              const RANSACOptions& options,
                                              AbsolutePoseWeightedRANSAC::Report* report) {
    // Scale the focal length by the given factor.
    Camera scaled_camera = camera;
    const std::vector<size_t>& focal_length_idxs = camera.FocalLengthIdxs();
    for (const size_t idx : focal_length_idxs) {
        scaled_camera.Params(idx) *= focal_length_factor;
    }

    // Normalize image coordinates with current camera hypothesis.
    std::vector<Eigen::Vector2d> points2D_N(points2D.size());
    for (size_t i = 0; i < points2D.size(); ++i) {
        points2D_N[i] = scaled_camera.ImageToWorld(points2D[i]);
    }

    // Estimate pose for given focal length.
    auto custom_options = options;
    custom_options.max_error =
            scaled_camera.ImageToWorldThreshold(options.max_error);
    AbsolutePoseWeightedRANSAC ransac(custom_options);
    *report = ransac.Estimate(points2D_N, points3D, distribution);
}

void EstimateAbsolutePoseWeightedLORANSACKernel(const Camera& camera,
                                                const double focal_length_factor,
                                                const std::vector<Eigen::Vector2d>& points2D,
                                                const std::vector<Eigen::Vector3d>& points3D,
                                                const std::vector<double>& distribution,
                                                const RANSACOptions& options,
                                                AbsolutePoseWeightedLORANSAC::Report* report) {
    // Scale the focal length by the given factor.
    Camera scaled_camera = camera;
    const std::vector<size_t>& focal_length_idxs = camera.FocalLengthIdxs();
    for (const size_t idx : focal_length_idxs) {
        scaled_camera.Params(idx) *= focal_length_factor;
    }

    // Normalize image coordinates with current camera hypothesis.
    std::vector<Eigen::Vector2d> points2D_N(points2D.size());
    for (size_t i = 0; i < points2D.size(); ++i) {
        points2D_N[i] = scaled_camera.ImageToWorld(points2D[i]);
    }

    // Estimate pose for given focal length.
    auto custom_options = options;
    custom_options.max_error =
            scaled_camera.ImageToWorldThreshold(options.max_error);
    AbsolutePoseWeightedLORANSAC ransac(custom_options);
    *report = ransac.Estimate(points2D_N, points3D, distribution);
}


void EstimateAbsolutePosePROSACKernel(const Camera& camera,
                                      const double focal_length_factor,
                                      const std::vector<Eigen::Vector2d>& points2D,
                                      const std::vector<Eigen::Vector3d>& points3D,
                                      const RANSACOptions& options,
                                      AbsolutePosePROSAC::Report* report) {
    // Scale the focal length by the given factor.
    Camera scaled_camera = camera;
    const std::vector<size_t>& focal_length_idxs = camera.FocalLengthIdxs();
    for (const size_t idx : focal_length_idxs) {
        scaled_camera.Params(idx) *= focal_length_factor;
    }

    // Normalize image coordinates with current camera hypothesis.
    std::vector<Eigen::Vector2d> points2D_N(points2D.size());
    for (size_t i = 0; i < points2D.size(); ++i) {
        points2D_N[i] = scaled_camera.ImageToWorld(points2D[i]);
    }

    // Estimate pose for given focal length.
    auto custom_options = options;
    custom_options.max_error =
            scaled_camera.ImageToWorldThreshold(options.max_error);
    AbsolutePosePROSAC ransac(custom_options);
    *report = ransac.Estimate(points2D_N, points3D);
}

void EstimateAbsolutePoseLOPROSACKernel(const Camera& camera,
                                        const double focal_length_factor,
                                        const std::vector<Eigen::Vector2d>& points2D,
                                        const std::vector<Eigen::Vector3d>& points3D,
                                        const RANSACOptions& options,
                                        AbsolutePoseLOPROSAC::Report* report) {
    // Scale the focal length by the given factor.
    Camera scaled_camera = camera;
    const std::vector<size_t>& focal_length_idxs = camera.FocalLengthIdxs();
    for (const size_t idx : focal_length_idxs) {
        scaled_camera.Params(idx) *= focal_length_factor;
    }

    // Normalize image coordinates with current camera hypothesis.
    std::vector<Eigen::Vector2d> points2D_N(points2D.size());
    for (size_t i = 0; i < points2D.size(); ++i) {
        points2D_N[i] = scaled_camera.ImageToWorld(points2D[i]);
    }

    // Estimate pose for given focal length.
    auto custom_options = options;
    custom_options.max_error =
            scaled_camera.ImageToWorldThreshold(options.max_error);
    AbsolutePoseLOPROSAC ransac(custom_options);
    *report = ransac.Estimate(points2D_N, points3D);
}

}  // namespace


/**==========================seg line=========================================*/

bool EstimateAbsolutePoseRANSAC(const AbsolutePoseEstimationOptions& options,
                               const std::vector<Eigen::Vector2d>& points2D,
                               const std::vector<Eigen::Vector3d>& points3D,
                               Eigen::Vector4d* qvec, Eigen::Vector3d* tvec,
                               Camera* camera, size_t* num_inliers,
                               std::vector<char>* inlier_mask) {
    options.Check();

    std::vector<double> focal_length_factors;
    if (options.estimate_focal_length) {
        // Generate focal length factors using a quadratic function,
        // such that more samples are drawn for small focal lengths
        focal_length_factors.reserve(options.num_focal_length_samples + 1);
        const double fstep = 1.0 / options.num_focal_length_samples;
        const double fscale =
                options.max_focal_length_ratio - options.min_focal_length_ratio;
        for (double f = 0; f <= 1.0; f += fstep) {
            focal_length_factors.push_back(options.min_focal_length_ratio +
                                           fscale * f * f);
        }
    } else {
        focal_length_factors.reserve(1);
        focal_length_factors.push_back(1);
    }

    std::vector<std::future<void>> futures;
    futures.resize(focal_length_factors.size());
    std::vector<typename AbsolutePoseRANSAC::Report,
            Eigen::aligned_allocator<typename AbsolutePoseRANSAC::Report>>
            reports;
    reports.resize(focal_length_factors.size());

    ThreadPool thread_pool(std::min(
            options.num_threads, static_cast<int>(focal_length_factors.size())));

    for (size_t i = 0; i < focal_length_factors.size(); ++i) {
        futures[i] = thread_pool.AddTask(
                EstimateAbsolutePoseRANSACKernel, *camera, focal_length_factors[i], points2D,
                points3D, options.ransac_options, &reports[i]);
    }

    double focal_length_factor = 0;
    Eigen::Matrix3x4d proj_matrix;
    *num_inliers = 0;
    inlier_mask->clear();

    // Find best model among all focal lengths.
    for (size_t i = 0; i < focal_length_factors.size(); ++i) {
        futures[i].get();
        const auto report = reports[i];
        if (report.success && report.support.num_inliers > *num_inliers) {
            *num_inliers = report.support.num_inliers;
            proj_matrix = report.model;
            *inlier_mask = report.inlier_mask;
            focal_length_factor = focal_length_factors[i];
        }
    }
    // std::cout << focal_length_factor << std::endl;
    if (*num_inliers == 0) {
        return false;
    }

    // Scale output camera with best estimated focal length.
    if (options.estimate_focal_length && *num_inliers > 0) {
        const std::vector<size_t>& focal_length_idxs = camera->FocalLengthIdxs();
        for (const size_t idx : focal_length_idxs) {
            camera->Params(idx) *= focal_length_factor;
        }
    }

    // Extract pose parameters.
    *qvec = RotationMatrixToQuaternion(proj_matrix.leftCols<3>());
    *tvec = proj_matrix.rightCols<1>();

    if (IsNaN(*qvec) || IsNaN(*tvec)) {
        return false;
    }

    return true;
}

bool EstimateAbsolutePoseLORANSAC(const AbsolutePoseEstimationOptions& options,
                                  const std::vector<Eigen::Vector2d>& points2D,
                                  const std::vector<Eigen::Vector3d>& points3D,
                                  Eigen::Vector4d* qvec, Eigen::Vector3d* tvec,
                                  Camera* camera, size_t* num_inliers,
                                  std::vector<char>* inlier_mask) {
    options.Check();

    std::vector<double> focal_length_factors;
    if (options.estimate_focal_length) {
        // Generate focal length factors using a quadratic function,
        // such that more samples are drawn for small focal lengths
        focal_length_factors.reserve(options.num_focal_length_samples + 1);
        const double fstep = 1.0 / options.num_focal_length_samples;
        const double fscale =
                options.max_focal_length_ratio - options.min_focal_length_ratio;
        for (double f = 0; f <= 1.0; f += fstep) {
            focal_length_factors.push_back(options.min_focal_length_ratio +
                                           fscale * f * f);
        }
    } else {
        focal_length_factors.reserve(1);
        focal_length_factors.push_back(1);
    }

    std::vector<std::future<void>> futures;
    futures.resize(focal_length_factors.size());
    std::vector<typename AbsolutePoseLORANSAC::Report,
            Eigen::aligned_allocator<typename AbsolutePoseLORANSAC::Report>>
            reports;
    reports.resize(focal_length_factors.size());

    ThreadPool thread_pool(std::min(
            options.num_threads, static_cast<int>(focal_length_factors.size())));

    for (size_t i = 0; i < focal_length_factors.size(); ++i) {
        futures[i] = thread_pool.AddTask(
                EstimateAbsolutePoseLORANSACKernel, *camera, focal_length_factors[i], points2D,
                points3D, options.ransac_options, &reports[i]);
    }

    double focal_length_factor = 0;
    Eigen::Matrix3x4d proj_matrix;
    *num_inliers = 0;
    inlier_mask->clear();

    // Find best model among all focal lengths.
    for (size_t i = 0; i < focal_length_factors.size(); ++i) {
        futures[i].get();
        const auto report = reports[i];
        if (report.success && report.support.num_inliers > *num_inliers) {
            *num_inliers = report.support.num_inliers;
            proj_matrix = report.model;
            *inlier_mask = report.inlier_mask;
            focal_length_factor = focal_length_factors[i];
        }
    }
    // std::cout << focal_length_factor << std::endl;
    if (*num_inliers == 0) {
        return false;
    }

    // Scale output camera with best estimated focal length.
    if (options.estimate_focal_length && *num_inliers > 0) {
        const std::vector<size_t>& focal_length_idxs = camera->FocalLengthIdxs();
        for (const size_t idx : focal_length_idxs) {
            camera->Params(idx) *= focal_length_factor;
        }
    }

    // Extract pose parameters.
    *qvec = RotationMatrixToQuaternion(proj_matrix.leftCols<3>());
    *tvec = proj_matrix.rightCols<1>();

    if (IsNaN(*qvec) || IsNaN(*tvec)) {
        return false;
    }

    return true;
}

bool EstimateAbsolutePoseWeightedRANSAC(const AbsolutePoseEstimationOptions &options,
                                        const std::vector<Eigen::Vector2d> &points2D,
                                        const std::vector<Eigen::Vector3d> &points3D,
                                        const std::vector<double> &distribution,
                                        Eigen::Vector4d *qvec, Eigen::Vector3d *tvec,
                                        Camera *camera, size_t *num_inliers,
                                        std::vector<char> *inlier_mask) {
    options.Check();

    std::vector<double> focal_length_factors;
    if (options.estimate_focal_length) {
        // Generate focal length factors using a quadratic function,
        // such that more samples are drawn for small focal lengths
        focal_length_factors.reserve(options.num_focal_length_samples + 1);
        const double fstep = 1.0 / options.num_focal_length_samples;
        const double fscale =
                options.max_focal_length_ratio - options.min_focal_length_ratio;
        for (double f = 0; f <= 1.0; f += fstep) {
            focal_length_factors.push_back(options.min_focal_length_ratio +
                                           fscale * f * f);
        }
    } else {
        focal_length_factors.reserve(1);
        focal_length_factors.push_back(1);
    }

    std::vector<std::future<void>> futures;
    futures.resize(focal_length_factors.size());
    std::vector<typename AbsolutePoseWeightedRANSAC::Report,
            Eigen::aligned_allocator<typename AbsolutePoseWeightedRANSAC::Report>>
            reports;
    reports.resize(focal_length_factors.size());

    ThreadPool thread_pool(std::min(
            options.num_threads, static_cast<int>(focal_length_factors.size())));

    for (size_t i = 0; i < focal_length_factors.size(); ++i) {
        futures[i] = thread_pool.AddTask(
                EstimateAbsolutePoseWeightedRANSACKernel, *camera, focal_length_factors[i], points2D,
                points3D, distribution, options.ransac_options, &reports[i]);
    }

    double focal_length_factor = 0;
    Eigen::Matrix3x4d proj_matrix;
    *num_inliers = 0;
    inlier_mask->clear();

    // Find best model among all focal lengths.
    for (size_t i = 0; i < focal_length_factors.size(); ++i) {
        futures[i].get();
        const auto report = reports[i];
        if (report.success && report.support.num_inliers > *num_inliers) {
            *num_inliers = report.support.num_inliers;
            proj_matrix = report.model;
            *inlier_mask = report.inlier_mask;
            focal_length_factor = focal_length_factors[i];
        }
    }
    // std::cout << focal_length_factor << std::endl;
    if (*num_inliers == 0) {
        return false;
    }

    // Scale output camera with best estimated focal length.
    if (options.estimate_focal_length && *num_inliers > 0) {
        const std::vector<size_t>& focal_length_idxs = camera->FocalLengthIdxs();
        for (const size_t idx : focal_length_idxs) {
            camera->Params(idx) *= focal_length_factor;
        }
    }

    // Extract pose parameters.
    *qvec = RotationMatrixToQuaternion(proj_matrix.leftCols<3>());
    *tvec = proj_matrix.rightCols<1>();

    if (IsNaN(*qvec) || IsNaN(*tvec)) {
        return false;
    }

    return true;
}

bool EstimateAbsolutePoseWeightedLORANSAC(const AbsolutePoseEstimationOptions &options,
                                          const std::vector<Eigen::Vector2d> &points2D,
                                          const std::vector<Eigen::Vector3d> &points3D,
                                          const std::vector<double> &distribution,
                                          Eigen::Vector4d *qvec, Eigen::Vector3d *tvec,
                                          Camera *camera, size_t *num_inliers,
                                          std::vector<char> *inlier_mask) {
    options.Check();

    std::vector<double> focal_length_factors;
    if (options.estimate_focal_length) {
        // Generate focal length factors using a quadratic function,
        // such that more samples are drawn for small focal lengths
        focal_length_factors.reserve(options.num_focal_length_samples + 1);
        const double fstep = 1.0 / options.num_focal_length_samples;
        const double fscale =
                options.max_focal_length_ratio - options.min_focal_length_ratio;
        for (double f = 0; f <= 1.0; f += fstep) {
            focal_length_factors.push_back(options.min_focal_length_ratio +
                                           fscale * f * f);
        }
    } else {
        focal_length_factors.reserve(1);
        focal_length_factors.push_back(1);
    }

    std::vector<std::future<void>> futures;
    futures.resize(focal_length_factors.size());
    std::vector<typename AbsolutePoseWeightedLORANSAC::Report,
            Eigen::aligned_allocator<typename AbsolutePoseWeightedLORANSAC::Report>>
            reports;
    reports.resize(focal_length_factors.size());

    ThreadPool thread_pool(std::min(
            options.num_threads, static_cast<int>(focal_length_factors.size())));

    for (size_t i = 0; i < focal_length_factors.size(); ++i) {
        futures[i] = thread_pool.AddTask(
                EstimateAbsolutePoseWeightedLORANSACKernel, *camera, focal_length_factors[i], points2D,
                points3D, distribution, options.ransac_options, &reports[i]);
    }

    double focal_length_factor = 0;
    Eigen::Matrix3x4d proj_matrix;
    *num_inliers = 0;
    inlier_mask->clear();

    // Find best model among all focal lengths.
    for (size_t i = 0; i < focal_length_factors.size(); ++i) {
        futures[i].get();
        const auto report = reports[i];
        if (report.success && report.support.num_inliers > *num_inliers) {
            *num_inliers = report.support.num_inliers;
            proj_matrix = report.model;
            *inlier_mask = report.inlier_mask;
            focal_length_factor = focal_length_factors[i];
        }
    }
    // std::cout << focal_length_factor << std::endl;
    if (*num_inliers == 0) {
        return false;
    }

    // Scale output camera with best estimated focal length.
    if (options.estimate_focal_length && *num_inliers > 0) {
        const std::vector<size_t>& focal_length_idxs = camera->FocalLengthIdxs();
        for (const size_t idx : focal_length_idxs) {
            camera->Params(idx) *= focal_length_factor;
        }
    }

    // Extract pose parameters.
    *qvec = RotationMatrixToQuaternion(proj_matrix.leftCols<3>());
    *tvec = proj_matrix.rightCols<1>();

    if (IsNaN(*qvec) || IsNaN(*tvec)) {
        return false;
    }

    return true;
}


bool EstimateAbsolutePosePROSAC(const AbsolutePoseEstimationOptions& options,
                                const std::vector<Eigen::Vector2d>& points2D,
                                const std::vector<Eigen::Vector3d>& points3D,
                                Eigen::Vector4d* qvec, Eigen::Vector3d* tvec,
                                Camera* camera, size_t* num_inliers,
                                std::vector<char>* inlier_mask) {
    options.Check();

    std::vector<double> focal_length_factors;
    if (options.estimate_focal_length) {
        // Generate focal length factors using a quadratic function,
        // such that more samples are drawn for small focal lengths
        focal_length_factors.reserve(options.num_focal_length_samples + 1);
        const double fstep = 1.0 / options.num_focal_length_samples;
        const double fscale =
                options.max_focal_length_ratio - options.min_focal_length_ratio;
        for (double f = 0; f <= 1.0; f += fstep) {
            focal_length_factors.push_back(options.min_focal_length_ratio +
                                           fscale * f * f);
        }
    } else {
        focal_length_factors.reserve(1);
        focal_length_factors.push_back(1);
    }

    std::vector<std::future<void>> futures;
    futures.resize(focal_length_factors.size());
    std::vector<typename AbsolutePosePROSAC::Report,
            Eigen::aligned_allocator<typename AbsolutePosePROSAC::Report>>
            reports;
    reports.resize(focal_length_factors.size());

    ThreadPool thread_pool(std::min(
            options.num_threads, static_cast<int>(focal_length_factors.size())));

    for (size_t i = 0; i < focal_length_factors.size(); ++i) {
        futures[i] = thread_pool.AddTask(
                EstimateAbsolutePosePROSACKernel, *camera, focal_length_factors[i], points2D,
                points3D, options.ransac_options, &reports[i]);
    }

    double focal_length_factor = 0;
    Eigen::Matrix3x4d proj_matrix;
    *num_inliers = 0;
    inlier_mask->clear();

    // Find best model among all focal lengths.
    for (size_t i = 0; i < focal_length_factors.size(); ++i) {
        futures[i].get();
        const auto report = reports[i];
        if (report.success && report.support.num_inliers > *num_inliers) {
            *num_inliers = report.support.num_inliers;
            proj_matrix = report.model;
            *inlier_mask = report.inlier_mask;
            focal_length_factor = focal_length_factors[i];
        }
    }
    // std::cout << focal_length_factor << std::endl;
    if (*num_inliers == 0) {
        return false;
    }

    // Scale output camera with best estimated focal length.
    if (options.estimate_focal_length && *num_inliers > 0) {
        const std::vector<size_t>& focal_length_idxs = camera->FocalLengthIdxs();
        for (const size_t idx : focal_length_idxs) {
            camera->Params(idx) *= focal_length_factor;
        }
    }

    // Extract pose parameters.
    *qvec = RotationMatrixToQuaternion(proj_matrix.leftCols<3>());
    *tvec = proj_matrix.rightCols<1>();

    if (IsNaN(*qvec) || IsNaN(*tvec)) {
        return false;
    }

    return true;
}

bool EstimateAbsolutePoseLOPROSAC(const AbsolutePoseEstimationOptions& options,
                                  const std::vector<Eigen::Vector2d>& points2D,
                                  const std::vector<Eigen::Vector3d>& points3D,
                                  Eigen::Vector4d* qvec, Eigen::Vector3d* tvec,
                                  Camera* camera, size_t* num_inliers,
                                  std::vector<char>* inlier_mask) {
    options.Check();

    std::vector<double> focal_length_factors;
    if (options.estimate_focal_length) {
        // Generate focal length factors using a quadratic function,
        // such that more samples are drawn for small focal lengths
        focal_length_factors.reserve(options.num_focal_length_samples + 1);
        const double fstep = 1.0 / options.num_focal_length_samples;
        const double fscale =
                options.max_focal_length_ratio - options.min_focal_length_ratio;
        for (double f = 0; f <= 1.0; f += fstep) {
            focal_length_factors.push_back(options.min_focal_length_ratio +
                                           fscale * f * f);
        }
    } else {
        focal_length_factors.reserve(1);
        focal_length_factors.push_back(1);
    }

    std::vector<std::future<void>> futures;
    futures.resize(focal_length_factors.size());
    std::vector<typename AbsolutePoseLOPROSAC::Report,
            Eigen::aligned_allocator<typename AbsolutePoseLOPROSAC::Report>>
            reports;
    reports.resize(focal_length_factors.size());

    ThreadPool thread_pool(std::min(
            options.num_threads, static_cast<int>(focal_length_factors.size())));

    for (size_t i = 0; i < focal_length_factors.size(); ++i) {
        futures[i] = thread_pool.AddTask(
                EstimateAbsolutePoseLOPROSACKernel, *camera, focal_length_factors[i], points2D,
                points3D, options.ransac_options, &reports[i]);
    }

    double focal_length_factor = 0;
    Eigen::Matrix3x4d proj_matrix;
    *num_inliers = 0;
    inlier_mask->clear();

    // Find best model among all focal lengths.
    for (size_t i = 0; i < focal_length_factors.size(); ++i) {
        futures[i].get();
        const auto report = reports[i];
        if (report.success && report.support.num_inliers > *num_inliers) {
            *num_inliers = report.support.num_inliers;
            proj_matrix = report.model;
            *inlier_mask = report.inlier_mask;
            focal_length_factor = focal_length_factors[i];
        }
    }
    // std::cout << focal_length_factor << std::endl;
    if (*num_inliers == 0) {
        return false;
    }

    // Scale output camera with best estimated focal length.
    if (options.estimate_focal_length && *num_inliers > 0) {
        const std::vector<size_t>& focal_length_idxs = camera->FocalLengthIdxs();
        for (const size_t idx : focal_length_idxs) {
            camera->Params(idx) *= focal_length_factor;
        }
    }

    // Extract pose parameters.
    *qvec = RotationMatrixToQuaternion(proj_matrix.leftCols<3>());
    *tvec = proj_matrix.rightCols<1>();

    if (IsNaN(*qvec) || IsNaN(*tvec)) {
        return false;
    }

    return true;
}


bool RefineAbsolutePose(const AbsolutePoseRefinementOptions& options,
                        const std::vector<char>& inlier_mask,
                        const std::vector<Eigen::Vector2d>& points2D,
                        const std::vector<Eigen::Vector3d>& points3D,
                        Eigen::Vector4d* qvec, Eigen::Vector3d* tvec,
                        Camera* camera) {
    CHECK_EQ(inlier_mask.size(), points2D.size());
    CHECK_EQ(points2D.size(), points3D.size());
    options.Check();

    ceres::LossFunction* loss_function =
            new ceres::CauchyLoss(options.loss_function_scale);

    double* camera_params_data = camera->ParamsData();
    double* qvec_data = qvec->data();
    double* tvec_data = tvec->data();

    std::vector<Eigen::Vector3d> points3D_copy = points3D;

    ceres::Problem problem;

    for (size_t i = 0; i < points2D.size(); ++i) {
        // Skip outlier observations
        if (!inlier_mask[i]) {
            continue;
        }

        ceres::CostFunction* cost_function = nullptr;

        switch (camera->ModelId()) {
#define CAMERA_MODEL_CASE(CameraModel)                                  \
case CameraModel::kModelId:                                           \
cost_function =                                                     \
    BundleAdjustmentCostFunction<CameraModel>::Create(points2D[i]); \
break;

            CAMERA_MODEL_SWITCH_CASES

#undef CAMERA_MODEL_CASE
        }

        problem.AddResidualBlock(cost_function, loss_function, qvec_data, tvec_data,
                                 points3D_copy[i].data(), camera_params_data);
        problem.SetParameterBlockConstant(points3D_copy[i].data());
    }

    if (problem.NumResiduals() > 0) {
        // Quaternion parameterization.
        *qvec = NormalizeQuaternion(*qvec);
        ceres::LocalParameterization* quaternion_parameterization =
                new ceres::QuaternionParameterization;
        problem.SetParameterization(qvec_data, quaternion_parameterization);

        // Camera parameterization.
        if (!options.refine_focal_length && !options.refine_extra_params) {
            problem.SetParameterBlockConstant(camera->ParamsData());
        } else {
            // Always set the principal point as fixed.
            std::vector<int> camera_params_const;
            const std::vector<size_t>& principal_point_idxs =
                    camera->PrincipalPointIdxs();
            camera_params_const.insert(camera_params_const.end(),
                                       principal_point_idxs.begin(),
                                       principal_point_idxs.end());

            if (!options.refine_focal_length) {
                const std::vector<size_t>& focal_length_idxs =
                        camera->FocalLengthIdxs();
                camera_params_const.insert(camera_params_const.end(),
                                           focal_length_idxs.begin(),
                                           focal_length_idxs.end());
            }

            if (!options.refine_extra_params) {
                const std::vector<size_t>& extra_params_idxs =
                        camera->ExtraParamsIdxs();
                camera_params_const.insert(camera_params_const.end(),
                                           extra_params_idxs.begin(),
                                           extra_params_idxs.end());
            }

            if (camera_params_const.size() == camera->NumParams()) {
                problem.SetParameterBlockConstant(camera->ParamsData());
            } else {
                ceres::SubsetParameterization* camera_params_parameterization =
                        new ceres::SubsetParameterization(
                                static_cast<int>(camera->NumParams()), camera_params_const);
                problem.SetParameterization(camera->ParamsData(),
                                            camera_params_parameterization);
            }
        }
    }

    ceres::Solver::Options solver_options;
    solver_options.gradient_tolerance = options.gradient_tolerance;
    solver_options.max_num_iterations = options.max_num_iterations;
    solver_options.linear_solver_type = ceres::DENSE_QR;

    // The overhead of creating threads is too large.
    solver_options.num_threads = 1;
#if CERES_VERSION_MAJOR < 2
    solver_options.num_linear_solver_threads = 1;
#endif  // CERES_VERSION_MAJOR

    ceres::Solver::Summary summary;
    ceres::Solve(solver_options, &problem, &summary);

    if (solver_options.minimizer_progress_to_stdout) {
        std::cout << std::endl;
    }

    if (options.print_summary) {
        PrintHeading2("Pose refinement report");
        PrintSolverSummary(summary);
    }

    return summary.IsSolutionUsable();
}

bool RefineAbsolutePoseWithPrior(const AbsolutePoseRefinementOptions& options,
                                const std::vector<char>& inlier_mask,
                                const std::vector<Eigen::Vector2d>& points2D,
                                const std::vector<Eigen::Vector3d>& points3D,
                                const std::vector<double>& priors,
                                Eigen::Vector4d* qvec, Eigen::Vector3d* tvec,
                                Camera* camera) {
    CHECK_EQ(inlier_mask.size(), points2D.size());
    CHECK_EQ(points2D.size(), points3D.size());
    options.Check();

    ceres::LossFunction* loss_function =
            new ceres::CauchyLoss(options.loss_function_scale);

    double* camera_params_data = camera->ParamsData();
    double* qvec_data = qvec->data();
    double* tvec_data = tvec->data();

    std::vector<Eigen::Vector3d> points3D_copy = points3D;
    std::vector<double> priors_copy = priors;

    ceres::Problem problem;

    for (size_t i = 0; i < points2D.size(); ++i) {
        // Skip outlier observations
        if (!inlier_mask[i]) {
            continue;
        }

        ceres::CostFunction* cost_function = nullptr;

        switch (camera->ModelId()) {
#define CAMERA_MODEL_CASE(CameraModel)                                  \
case CameraModel::kModelId:                                           \
cost_function =                                                     \
    BundleAdjustmentCostFunctionWithPrior<CameraModel>::Create(points2D[i]); \
break;

            CAMERA_MODEL_SWITCH_CASES

#undef CAMERA_MODEL_CASE
        }

        problem.AddResidualBlock(cost_function, loss_function, qvec_data, tvec_data,
                                 points3D_copy[i].data(), camera_params_data, &priors_copy[i]);
        problem.SetParameterBlockConstant(points3D_copy[i].data());
        problem.SetParameterBlockConstant(&priors_copy[i]);
    }

    if (problem.NumResiduals() > 0) {
        // Quaternion parameterization.
        *qvec = NormalizeQuaternion(*qvec);
        ceres::LocalParameterization* quaternion_parameterization =
                new ceres::QuaternionParameterization;
        problem.SetParameterization(qvec_data, quaternion_parameterization);

        // Camera parameterization.
        if (!options.refine_focal_length && !options.refine_extra_params) {
            problem.SetParameterBlockConstant(camera->ParamsData());
        } else {
            // Always set the principal point as fixed.
            std::vector<int> camera_params_const;
            const std::vector<size_t>& principal_point_idxs =
                    camera->PrincipalPointIdxs();
            camera_params_const.insert(camera_params_const.end(),
                                       principal_point_idxs.begin(),
                                       principal_point_idxs.end());

            if (!options.refine_focal_length) {
                const std::vector<size_t>& focal_length_idxs =
                        camera->FocalLengthIdxs();
                camera_params_const.insert(camera_params_const.end(),
                                           focal_length_idxs.begin(),
                                           focal_length_idxs.end());
            }

            if (!options.refine_extra_params) {
                const std::vector<size_t>& extra_params_idxs =
                        camera->ExtraParamsIdxs();
                camera_params_const.insert(camera_params_const.end(),
                                           extra_params_idxs.begin(),
                                           extra_params_idxs.end());
            }

            if (camera_params_const.size() == camera->NumParams()) {
                problem.SetParameterBlockConstant(camera->ParamsData());
            } else {
                ceres::SubsetParameterization* camera_params_parameterization =
                        new ceres::SubsetParameterization(
                                static_cast<int>(camera->NumParams()), camera_params_const);
                problem.SetParameterization(camera->ParamsData(),
                                            camera_params_parameterization);
            }
        }
    }

    ceres::Solver::Options solver_options;
    solver_options.gradient_tolerance = options.gradient_tolerance;
    solver_options.max_num_iterations = options.max_num_iterations;
    solver_options.linear_solver_type = ceres::DENSE_QR;

    // The overhead of creating threads is too large.
    solver_options.num_threads = 1;
#if CERES_VERSION_MAJOR < 2
    solver_options.num_linear_solver_threads = 1;
#endif  // CERES_VERSION_MAJOR

    ceres::Solver::Summary summary;
    ceres::Solve(solver_options, &problem, &summary);

    if (solver_options.minimizer_progress_to_stdout) {
        std::cout << std::endl;
    }

    if (options.print_summary) {
        PrintHeading2("Pose refinement report");
        PrintSolverSummary(summary);
    }

    return summary.IsSolutionUsable();
}


}
