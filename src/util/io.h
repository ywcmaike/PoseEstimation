//
// Created by yuhailin.
//

#ifndef VLOCTIO_IO_H
#define VLOCTIO_IO_H

#include <string>
#include <vector>
#include <unordered_map>

#include "util/types.h"

namespace PE {


void ReadCorresspondencesTxt(const std::string &path, std::vector<Eigen::Vector2d> &points2D,
                             std::vector<Eigen::Vector3d> &points3D);

void ReadCorresspondencesTxt(const std::string &path, std::vector<Eigen::Vector2d> &points2D,
                             std::vector<Eigen::Vector3d> &points3D,
                             std::vector<double> &priors);

void WriteCorrespondencesTxt(const std::string &path, const std::vector<std::pair<size_t, size_t>> &edges,
                             const std::vector<Eigen::Vector2d> &points2D, const std::vector<Eigen::Vector3d> &points3D);

}


#endif //VLOCTIO_IO_H
