//
// Created by yuhailin.
//

#ifndef VLOCTIO_BUNDLE_ADJUSTMENT_H
#define VLOCTIO_BUNDLE_ADJUSTMENT_H

#include <memory>
#include <unordered_set>
#include <unordered_map>

#include <Eigen/Core>

#include <ceres/ceres.h>

#include "util/alignment.h"
#include "util/types.h"

namespace PE {

struct BundleAdjustmentOptions {
    // Loss function types: Trivial (non-robust) and Cauchy (robust) loss.
    enum class LossFunctionType { TRIVIAL, SOFT_L1, CAUCHY };
    LossFunctionType loss_function_type = LossFunctionType::TRIVIAL;

    // Scaling factor determines residual at which robustification takes place.
    double loss_function_scale = 1.0;

    // Whether to refine the focal length parameter group.
    bool refine_focal_length = true;

    // Whether to refine the principal point parameter group.
    bool refine_principal_point = false;

    // Whether to refine the extra parameter group.
    bool refine_extra_params = true;

    // Whether to refine the extrinsic parameter group.
    bool refine_extrinsics = true;

    // Whether to print a final summary.
    bool print_summary = true;

    // Ceres-Solver options.
    ceres::Solver::Options solver_options;

    BundleAdjustmentOptions() {
        solver_options.function_tolerance = 0.0;
        solver_options.gradient_tolerance = 0.0;
        solver_options.parameter_tolerance = 0.0;
        solver_options.minimizer_progress_to_stdout = false;
        solver_options.max_num_iterations = 100;
        solver_options.max_linear_solver_iterations = 200;
        solver_options.max_num_consecutive_invalid_steps = 10;
        solver_options.max_consecutive_nonmonotonic_steps = 10;
        solver_options.num_threads = -1;
#if CERES_VERSION_MAJOR < 2
        solver_options.num_linear_solver_threads = -1;
#endif  // CERES_VERSION_MAJOR
    }

    // Create a new loss function based on the specified options. The caller
    // takes ownership of the loss function.
    ceres::LossFunction* CreateLossFunction() const;

    bool Check() const;
};

// Configuration container to setup bundle adjustment problems.
class BundleAdjustmentConfig {
public:
    BundleAdjustmentConfig();

    size_t NumImages() const;
    size_t NumPoints() const;
    size_t NumConstantCameras() const;
    size_t NumConstantPoses() const;
    size_t NumConstantTvecs() const;
    size_t NumVariablePoints() const;
    size_t NumConstantPoints() const;

    // Determine the number of residuals for the given reconstruction. The number
    // of residuals equals the number of observations times two.
//    size_t NumResiduals(const Reconstruction& reconstruction) const;

    // Add / remove images from the configuration.
    void AddImage(const image_t image_id);
    bool HasImage(const image_t image_id) const;
    void RemoveImage(const image_t image_id);

    // Set cameras of added images as constant or variable. By default all
    // cameras of added images are variable. Note that the corresponding images
    // have to be added prior to calling these methods.
    void SetConstantCamera(const camera_t camera_id);
    void SetVariableCamera(const camera_t camera_id);
    bool IsConstantCamera(const camera_t camera_id) const;

    // Set the pose of added images as constant. The pose is defined as the
    // rotational and translational part of the projection matrix.
    void SetConstantPose(const image_t image_id);
    void SetVariablePose(const image_t image_id);
    bool HasConstantPose(const image_t image_id) const;

    // Set the translational part of the pose, hence the constant pose
    // indices may be in [0, 1, 2] and must be unique. Note that the
    // corresponding images have to be added prior to calling these methods.
    void SetConstantTvec(const image_t image_id, const std::vector<int>& idxs);
    void RemoveConstantTvec(const image_t image_id);
    bool HasConstantTvec(const image_t image_id) const;

    // Add / remove points from the configuration. Note that points can either
    // be variable or constant but not both at the same time.
    void AddVariablePoint(const point3D_t point3D_id);
    void AddConstantPoint(const point3D_t point3D_id);
    bool HasPoint(const point3D_t point3D_id) const;
    bool HasVariablePoint(const point3D_t point3D_id) const;
    bool HasConstantPoint(const point3D_t point3D_id) const;
    void RemoveVariablePoint(const point3D_t point3D_id);
    void RemoveConstantPoint(const point3D_t point3D_id);

    // Access configuration data.
    const std::unordered_set<image_t>& Images() const;
    const std::unordered_set<point3D_t>& VariablePoints() const;
    const std::unordered_set<point3D_t>& ConstantPoints() const;
    const std::vector<int>& ConstantTvec(const image_t image_id) const;

private:
    std::unordered_set<camera_t> constant_camera_ids_;
    std::unordered_set<image_t> image_ids_;
    std::unordered_set<point3D_t> variable_point3D_ids_;
    std::unordered_set<point3D_t> constant_point3D_ids_;
    std::unordered_set<image_t> constant_poses_;
    std::unordered_map<image_t, std::vector<int>> constant_tvecs_;
};

// Bundle adjustment based on Ceres-Solver. Enables most flexible configurations
// and provides best solution quality.
class BundleAdjuster {
public:
    BundleAdjuster(const BundleAdjustmentOptions& options,
                   const BundleAdjustmentConfig& config);

//    bool Solve(Reconstruction* reconstruction);

    // Get the Ceres solver summary for the last call to `Solve`.
    const ceres::Solver::Summary& Summary() const;

private:
//    void SetUp(Reconstruction* reconstruction,
//               ceres::LossFunction* loss_function);
//    void TearDown(Reconstruction* reconstruction);

//    void AddImageToProblem(const image_t image_id, Reconstruction* reconstruction,
//                           ceres::LossFunction* loss_function);

//    void AddPointToProblem(const point3D_t point3D_id,
//                           Reconstruction* reconstruction,
//                           ceres::LossFunction* loss_function);

protected:
//    void ParameterizeCameras(Reconstruction* reconstruction);
//    void ParameterizePoints(Reconstruction* reconstruction);

    const BundleAdjustmentOptions options_;
    BundleAdjustmentConfig config_;
    std::unique_ptr<ceres::Problem> problem_;
    ceres::Solver::Summary summary_;
    std::unordered_set<camera_t> camera_ids_;
    std::unordered_map<point3D_t, size_t> point3D_num_observations_;
};

void PrintSolverSummary(const ceres::Solver::Summary& summary);

}

#endif //VLOCTIO_BUNDLE_ADJUSTMENT_H
