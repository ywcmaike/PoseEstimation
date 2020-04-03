//
// Created by yuhailin.
//

#include "base/projection.h"

#include "base/pose.h"
#include "util/matrix.h"

namespace PE {

Eigen::Matrix3x4d ComposeProjectionMatrix(const Eigen::Vector4d& qvec,
                                          const Eigen::Vector3d& tvec) {
    Eigen::Matrix3x4d proj_matrix;
    proj_matrix.leftCols<3>() = QuaternionToRotationMatrix(qvec);
    proj_matrix.rightCols<1>() = tvec;
    return proj_matrix;
}

Eigen::Matrix3x4d ComposeProjectionMatrix(const Eigen::Matrix3d& R,
                                          const Eigen::Vector3d& T) {
    Eigen::Matrix3x4d proj_matrix;
    proj_matrix.leftCols<3>() = R;
    proj_matrix.rightCols<1>() = T;
    return proj_matrix;
}

Eigen::Matrix3x4d InvertProjectionMatrix(const Eigen::Matrix3x4d& proj_matrix) {
    Eigen::Matrix3x4d inv_proj_matrix;
    inv_proj_matrix.leftCols<3>() = proj_matrix.leftCols<3>().transpose();
    inv_proj_matrix.rightCols<1>() = ProjectionCenterFromMatrix(proj_matrix);
    return inv_proj_matrix;
}

Eigen::Matrix3d ComputeClosestRotationMatrix(const Eigen::Matrix3d& matrix) {
    const Eigen::JacobiSVD<Eigen::Matrix3d> svd(
            matrix, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Matrix3d R = svd.matrixU() * (svd.matrixV().transpose());
    if (R.determinant() < 0.0) {
        R *= -1.0;
    }
    return R;
}

bool DecomposeProjectionMatrix(const Eigen::Matrix3x4d& P, Eigen::Matrix3d* K,
                               Eigen::Matrix3d* R, Eigen::Vector3d* T) {
    Eigen::Matrix3d RR;
    Eigen::Matrix3d QQ;
    DecomposeMatrixRQ(P.leftCols<3>().eval(), &RR, &QQ);

    *R = ComputeClosestRotationMatrix(QQ);

    const double det_K = RR.determinant();
    if (det_K == 0) {
        return false;
    } else if (det_K > 0) {
        *K = RR;
    } else {
        *K = -RR;
    }

    for (int i = 0; i < 3; ++i) {
        if ((*K)(i, i) < 0.0) {
            K->col(i) = -K->col(i);
            R->row(i) = -R->row(i);
        }
    }

    *T = K->triangularView<Eigen::Upper>().solve(P.col(3));
    if (det_K < 0) {
        *T = -(*T);
    }

    return true;
}

Eigen::Vector2d ProjectPointToImage(const Eigen::Vector3d& point3D,
                                    const Eigen::Matrix3x4d& proj_matrix,
                                    const Camera& camera) {
    const Eigen::Vector3d world_point = proj_matrix * point3D.homogeneous();
    return camera.WorldToImage(world_point.hnormalized());
}

double CalculateSquaredReprojectionError(const Eigen::Vector2d& point2D,
                                         const Eigen::Vector3d& point3D,
                                         const Eigen::Vector4d& qvec,
                                         const Eigen::Vector3d& tvec,
                                         const Camera& camera) {
    const Eigen::Vector3d proj_point3D =
            QuaternionRotatePoint(qvec, point3D) + tvec;

    // Check that point is infront of camera.
    if (proj_point3D.z() < std::numeric_limits<double>::epsilon()) {
        return std::numeric_limits<double>::max();
    }

    const Eigen::Vector2d proj_point2D =
            camera.WorldToImage(proj_point3D.hnormalized());

    return (proj_point2D - point2D).squaredNorm();
}

double CalculateSquaredReprojectionError(const Eigen::Vector2d& point2D,
                                         const Eigen::Vector3d& point3D,
                                         const Eigen::Matrix3x4d& proj_matrix,
                                         const Camera& camera) {
    const double proj_z = proj_matrix.row(2).dot(point3D.homogeneous());

    // Check that point is infront of camera.
    if (proj_z < std::numeric_limits<double>::epsilon()) {
        return std::numeric_limits<double>::max();
    }

    const double proj_x = proj_matrix.row(0).dot(point3D.homogeneous());
    const double proj_y = proj_matrix.row(1).dot(point3D.homogeneous());
    const double inv_proj_z = 1.0 / proj_z;

    const Eigen::Vector2d proj_point2D = camera.WorldToImage(
            Eigen::Vector2d(inv_proj_z * proj_x, inv_proj_z * proj_y));

    return (proj_point2D - point2D).squaredNorm();
}

double CalculateAngularError(const Eigen::Vector2d& point2D,
                             const Eigen::Vector3d& point3D,
                             const Eigen::Vector4d& qvec,
                             const Eigen::Vector3d& tvec,
                             const Camera& camera) {
    return CalculateNormalizedAngularError(camera.ImageToWorld(point2D), point3D,
                                           qvec, tvec);
}

double CalculateAngularError(const Eigen::Vector2d& point2D,
                             const Eigen::Vector3d& point3D,
                             const Eigen::Matrix3x4d& proj_matrix,
                             const Camera& camera) {
    return CalculateNormalizedAngularError(camera.ImageToWorld(point2D), point3D,
                                           proj_matrix);
}

double CalculateNormalizedAngularError(const Eigen::Vector2d& point2D,
                                       const Eigen::Vector3d& point3D,
                                       const Eigen::Vector4d& qvec,
                                       const Eigen::Vector3d& tvec) {
    const Eigen::Vector3d ray1 = point2D.homogeneous();
    const Eigen::Vector3d ray2 = QuaternionRotatePoint(qvec, point3D) + tvec;
    return std::acos(ray1.normalized().transpose() * ray2.normalized());
}

double CalculateNormalizedAngularError(const Eigen::Vector2d& point2D,
                                       const Eigen::Vector3d& point3D,
                                       const Eigen::Matrix3x4d& proj_matrix) {
    const Eigen::Vector3d ray1 = point2D.homogeneous();
    const Eigen::Vector3d ray2 = proj_matrix * point3D.homogeneous();
    return std::acos(ray1.normalized().transpose() * ray2.normalized());
}

double CalculateDepth(const Eigen::Matrix3x4d& proj_matrix,
                      const Eigen::Vector3d& point3D) {
    const double proj_z = proj_matrix.row(2).dot(point3D.homogeneous());
    return proj_z * proj_matrix.col(2).norm();
}

bool HasPointPositiveDepth(const Eigen::Matrix3x4d& proj_matrix,
                           const Eigen::Vector3d& point3D) {
    return proj_matrix.row(2).dot(point3D.homogeneous()) >=
           std::numeric_limits<double>::epsilon();
}

}  // namespace PE
