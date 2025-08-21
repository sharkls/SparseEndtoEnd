# Copyright (c) 2024 SparseEnd2End. All rights reserved @author: Thomas Von Wu.
cmake_minimum_required(VERSION 3.16.3)
project(SparseEnd2EndOnboard CUDA CXX)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_STANDARD 17)

set(CMAKE_CXX_FLAGS_RELEASE
    "-std=c++17 -Wextra -Wall -Wno-missing-field-initializers -Wno-deprecated-declarations -O3 -DENABLE_TEXT_BACKEND_STB"
)
set(CMAKE_CXX_FLAGS_DEBUG "-std=c++17 -O0 -g -DENABLE_TEXT_BACKEND_STB")

set(CUDA_NVCC_FLAGS_RELEASE
    "-Werror=all-warnings -Xcompiler -std=c++17,-Wextra,-Wall,-Wno-deprecated-declarations,-O3 -DENABLE_TEXT_BACKEND_STB"
)
set(CUDA_NVCC_FLAGS_DEBUG "-std=c++17 -O0 -g -DENABLE_TEXT_BACKEND_STB")

# CUDA setting
find_package(CUDA 11 REQUIRED)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_86,code=sm_86")
include_directories(${CUDA_INCLUDE_DIRS})
link_directories(${CUDA_TOOLKIT_ROOT_DIR}/lib64)

# TensorRT setting
set(ENV_TensorRT_INC /mnt/env/tensorrt/TensorRT-8.5.1.7/include)
set(ENV_TensorRT_LIB /mnt/env/tensorrt/TensorRT-8.5.1.7/lib)
set(ENV_cuDNN_LIB
    /mnt/env/tensorrt/cudnn-linux-x86_64-8.6.0.163_cuda11-archive/lib)
include_directories(${ENV_TensorRT_INC})
link_directories(${ENV_TensorRT_LIB} ${ENV_cuDNN_LIB})

include_directories(
  ${CMAKE_SOURCE_DIR}/common 
  ${CMAKE_SOURCE_DIR}/preprocessor
  ${CMAKE_SOURCE_DIR}/tensorrt)

set(Sparse4dOnboardBin sparse4d_onboard.bin)

# 添加编译源码
add_executable(
  ${Sparse4dOnboardBin}
  main.cpp
  ${CMAKE_SOURCE_DIR}/tensorrt/tensorrt.cpp
  ${CMAKE_SOURCE_DIR}/tensorrt/logging.cpp
  ${CMAKE_SOURCE_DIR}/preprocessor/parameters_parser.cpp)

# 链接三方库
target_link_libraries(
    ${Sparse4dOnboardBin}
    gtest
    gtest_main
    gmock_main
    pthread
    dl
    cudart
    cudnn
    yaml-cpp
    nvinfer
    nvinfer_plugin
    nvonnxparser)

set(GTEST_SOURCE_DIR ${CMAKE_SOURCE_DIR}/third_party/googletest)
set(GTEST_BINARY_DIR ${CMAKE_BINARY_DIR}/third_party/googletest)
add_subdirectory(${GTEST_SOURCE_DIR} ${GTEST_BINARY_DIR})

set(YAMLCPP_SOURCE_DIR ${CMAKE_SOURCE_DIR}/third_party/yaml-cpp)
set(YAMLCPP_BINARY_DIR ${CMAKE_BINARY_DIR}/third_party/yaml-cpp)
add_subdirectory(${YAMLCPP_SOURCE_DIR} ${YAMLCPP_BINARY_DIR})



# include_directories($ENV{ENV_TensorRT_INC} $ENV{ENV_CUDA_INC})
# link_directories($ENV{ENV_TensorRT_LIB} $ENV{ENV_CUDA_LIB} build)

# add_subdirectory(common preprocessor)
# add_executable(main.bin main.cpp)
# target_link_libraries(main.bin PRIVATE common)
