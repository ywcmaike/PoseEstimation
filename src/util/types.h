//
// Created by yuhailin.
//

#ifndef VLOCTIO_TYPES_H
#define VLOCTIO_TYPES_H

#include "util/alignment.h"

#include <cstdint>

// Define non-copyable or non-movable classes.
#define NON_COPYABLE(class_name)          \
  class_name(class_name const&) = delete; \
  void operator=(class_name const& obj) = delete;
#define NON_MOVABLE(class_name) class_name(class_name&&) = delete;

#include <Eigen/Core>

namespace Eigen {

    typedef Eigen::Matrix<float, 3, 4> Matrix3x4f;
    typedef Eigen::Matrix<double, 3, 4> Matrix3x4d;
    typedef Eigen::Matrix<uint8_t, 3, 1> Vector3ub;
    typedef Eigen::Matrix<uint8_t, 4, 1> Vector4ub;
    typedef Eigen::Matrix<double, 6, 1> Vector6d;

}  // namespace Eigen

namespace PE {

////////////////////////////////////////////////////////////////////////////////
// Index types, determines the maximum number of objects.
////////////////////////////////////////////////////////////////////////////////

// Unique identifier for cameras.
    typedef uint32_t camera_t;

// Unique identifier for images.
    typedef uint32_t image_t;

// Each image pair gets a unique ID, see `Database::ImagePairToPairId`.
    typedef uint64_t image_pair_t;

// Index per image, i.e. determines maximum number of 2D points per image.
    typedef uint32_t point2D_t;

// Unique identifier per added 3D point. Since we add many 3D points,
// delete them, and possibly re-add them again, the maximum number of allowed
// unique indices should be large.
    typedef uint64_t point3D_t;

// Values for invalid identifiers or indices.
    const camera_t kInvalidCameraId = std::numeric_limits<camera_t>::max();
    const image_t kInvalidImageId = std::numeric_limits<image_t>::max();
    const image_pair_t kInvalidImagePairId =
            std::numeric_limits<image_pair_t>::max();
    const point2D_t kInvalidPoint2DIdx = std::numeric_limits<point2D_t>::max();
    const point3D_t kInvalidPoint3DId = std::numeric_limits<point3D_t>::max();

}  // namespace PE

// This file provides specializations of the templated hash function for
// custom types. These are used for comparison in unordered sets/maps.
namespace std {

// Hash function specialization for uint32_t pairs, e.g., image_t or camera_t.
    template <>
    struct hash<std::pair<uint32_t, uint32_t>> {
        std::size_t operator()(const std::pair<uint32_t, uint32_t>& p) const {
            const uint64_t s = (static_cast<uint64_t>(p.first) << 32) +
                               static_cast<uint64_t>(p.second);
            return std::hash<uint64_t>()(s);
        }
    };

}  // namespace std

#endif //VLOCTIO_TYPES_H
