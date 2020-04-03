//
// Created by yuhailin.
//

#include <iostream>
#include <fstream>

#include "util/io.h"
#include "util/logging.h"

namespace PE {


void ReadCorresspondencesTxt(const std::string &path, std::vector<Eigen::Vector2d> &points2D,
                             std::vector<Eigen::Vector3d> &points3D) {
    std::ifstream file(path);
    CHECK(file.is_open()) << path;

    int num_corresspondeces = 0;
    std::string line;

    getline(file, line);
    std::stringstream ss1(line);
    ss1 >> num_corresspondeces;

    points2D.resize(num_corresspondeces);
    points3D.resize(num_corresspondeces);
    for (size_t i = 0; i != num_corresspondeces; ++i) {
        getline(file, line);
        std::stringstream ss(line);
        ss >> points2D[i][0] >> points2D[i][1]
           >> points3D[i][0] >> points3D[i][1] >> points3D[i][2];
    }
}

void ReadCorresspondencesTxt(const std::string &path, std::vector<Eigen::Vector2d> &points2D,
                             std::vector<Eigen::Vector3d> &points3D,
                             std::vector<double> &priors) {
    std::ifstream file(path);
    CHECK(file.is_open()) << path;

    int num_corresspondeces = 0;
    std::string line;

    getline(file, line);
    std::stringstream ss1(line);
    ss1 >> num_corresspondeces;

    points2D.resize(num_corresspondeces);
    points3D.resize(num_corresspondeces);
    priors.resize(num_corresspondeces);
    for (size_t i = 0; i != num_corresspondeces; ++i) {
        getline(file, line);
        std::stringstream ss(line);
        ss >> points2D[i][0] >> points2D[i][1]
           >> points3D[i][0] >> points3D[i][1] >> points3D[i][2]
           >> priors[i];
        priors[i] = priors[i]*sqrt(2*3.141592653589793);
    }
}

void WriteCorrespondencesTxt(const std::string &path, const std::vector<std::pair<size_t, size_t>> &edges,
                             const std::vector<Eigen::Vector2d> &points2D, const std::vector<Eigen::Vector3d> &points3D){
    std::ofstream file(path);
    CHECK(file.is_open()) << path;

    file << points2D.size() << std::endl;
    for (size_t i = 0; i != points2D.size(); ++i) {
        file << points2D[i][0] << " " << points2D[i][1] << std::endl;
    }

    file << points3D.size() << std::endl;
    for (size_t i = 0; i != points3D.size(); ++i) {
        file << points3D[i][0] << " " << points3D[i][1] << " " << points3D[i][2] << std::endl;
    }

    file << edges.size() << std::endl;
    for (size_t i = 0; i != edges.size(); ++i) {
        file << edges[i].first << " " << edges[i].second << std::endl;
    }
    file.close();
}


}
