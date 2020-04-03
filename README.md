# Pose Estimation
This a pose estimation program. The input is 2d-3d correspondences and output 6 Dof camera pose.

## Dependencies
- c++11
- [CMake](https://cmake.org/) is a cross platform build system.
- [Eigen3](http://eigen.tuxfamily.org/index.php?title=Main_Page) is used extensively for doing nearly all the matrix and linear algebra operations.
- [OpenCV](https://github.com/opencv/opencv) is an Open Source Computer Vision Library, just for visualization.
- [Ceres](http://ceres-solver.org/) solver is an open source C++ library for modeling and solving large, complicated optimization problems.


## Usage
```
# Build Project
mkdir build
cmake ..
make -j
```