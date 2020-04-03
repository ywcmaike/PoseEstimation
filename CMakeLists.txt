# Copyright (c) 2020, Hailin Yu.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#     * Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#
#     * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#
# Author: Hailin Yu
cmake_minimum_required(VERSION 3.12)


set(CMAKE_CXX_STANDARD 11)


if(IS_MSVC)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /W3")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W3")
elseif(IS_GNU OR IS_CLANG)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall")
endif()

# include
include_directories(.)
include_directories(./src)

# Eigen
find_package(Eigen3 REQUIRED)
if(EIGEN3_FOUND)
    include_directories(${EIGEN3_INCLUDE_DIRS})
    message( STATUS "EIGEN3_INCLUDE_DIRS = ${EIGEN3_INCLUDE_DIRS}")
endif()

# Boost
set(Boost_USE_STATIC_LIBS OFF)
set(Boost_USE_MULTITHREADED ON)
set(Boost_USE_STATIC_RUNTIME OFF)
find_package(Boost 1.45.0 COMPONENTS filesystem system)
if(Boost_FOUND)
    include_directories(${Boost_INCLUDE_DIRS})
endif()

# OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
link_directories(${OpenCV_LIBRARY_DIRS})
if(OpenCV_FOUND)
    message( STATUS "OpenCV_INCLUDE_DIRS = ${OpenCV_INCLUDE_DIRS}")
    message( STATUS "OpenCV_LIBRARIES = ${OpenCV_LIBRARIES}")
    message( STATUS "OpenCV_LIB_VERSION = ${OpenCV_LIB_VERSION}")
endif()

# Ceres
find_package(Ceres REQUIRED)
if(Ceres_FOUND)
    include_directories(${Ceres_INCLUDE_DIRS})
    message( STATUS "Ceres_INCLUDE_DIRS = ${CERES_INCLUDE_DIRS}")
    message( STATUS "Ceres_LIBRARIES = ${Ceres_LIBRARIES}")
    message( STATUS "Ceres_LIB_VERSION = ${Ceres_LIB_VERSION}")
endif()

set(base_src
    src/base/camera.cc
    src/base/camera_models.cc
    src/base/pose.cc
    src/base/polynomial.cc
    src/base/projection.cc
    )

set(estimators_src
    src/estimators/absolute_pose.cc
    src/estimators/utils.cc
    src/estimators/pose.cc
    src/estimators/random_sampler.cc
    src/estimators/support_measurement.cc
    )

set(pose_src
    src/pose/pose_estimation.cc
    )


set(optim_src
    src/optim/bundle_adjustment.cc
    )

set(util_src
    src/util/io.cc
    src/util/logging.cc
    src/util/misc.cc
    src/util/string.cc
    src/util/math.cc
    src/util/threading.cc
    src/util/timer.cc
    src/util/random.cc
    )

add_executable(pose_estimation exe/solve_pose_benchmark.cc ${base_src} ${util_src} ${estimators_src} ${optim_src} ${pose_src})

target_link_libraries(pose_estimation ${OpenCV_LIBS}  ${CERES_LIBRARIES}  ${Boost_LIBRARIES})

