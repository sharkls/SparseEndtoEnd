# Copyright (c) 2024 SparseEnd2End. All rights reserved @author: Thomas Von Wu.

cmake_minimum_required(VERSION 3.16)

# 设置项目
project(dfa_plugin CUDA CXX)

# 启用CUDA语言支持
enable_language(CUDA)

# 设置C++标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 设置编译选项
option(DEBUG "Enable debug mode" OFF)
option(WARN "Enable warnings" OFF)

# 设置CUDA架构
# set(CUDASM "75" CACHE STRING "CUDA architecture")  # CUDA11.6 
set(CUDASM "86" CACHE STRING "CUDA architecture")  # CUDA12.4
set(CMAKE_CUDA_ARCHITECTURES ${CUDASM})

# 设置编译标志
if(DEBUG)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O0")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -g -G")
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3")
endif()

if(WARN)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wunused-function -Wunused-variable -Wfatal-errors")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Wall -Wunused-function -Wunused-variable -Wfatal-errors")
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -w")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -w")
endif()

# 添加共享库编译选项
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --shared -Xcompiler -fPIC")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")

# 设置包含目录
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
    $ENV{TENSORRT_HOME}/include
)

# 设置库目录
link_directories(
    ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES}
    $ENV{TENSORRT_HOME}/lib
    $ENV{CUDNN_HOME}/lib
)

# 收集源文件
file(GLOB CXX_SOURCES "*.cpp")
file(GLOB CUDA_SOURCES "*.cu")
file(GLOB HEADER_FILES "*.h")

# 设置输出目录和文件名
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/lib)

# 创建共享库
add_library(deformableAttentionAggr SHARED
    ${CXX_SOURCES}
    ${CUDA_SOURCES}
    ${HEADER_FILES}
)

# 设置目标属性
set_target_properties(deformableAttentionAggr PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
    OUTPUT_NAME "deformableAttentionAggr"
    PREFIX ""
    SUFFIX ".so"
)

# 链接库
target_link_libraries(deformableAttentionAggr
    cudart
    cublas
    nvinfer
    nvonnxparser
    cudnn
    stdc++fs
)

# 添加自定义目标用于显示信息
add_custom_target(show
    COMMAND ${CMAKE_COMMAND} -E echo "Include directories: ${CMAKE_CXX_FLAGS}"
    COMMAND ${CMAKE_COMMAND} -E echo "Library directories: ${CMAKE_CUDA_FLAGS}"
    COMMAND ${CMAKE_COMMAND} -E echo "Source files: ${CXX_SOURCES} ${CUDA_SOURCES}"
)

# 添加清理目标
add_custom_target(clean_all
    COMMAND ${CMAKE_COMMAND} -E remove_directory build
    COMMAND ${CMAKE_COMMAND} -E remove_directory lib
)