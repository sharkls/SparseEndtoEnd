project(SparseBEVAlg8.6)

message("CMAKE_SOURCE_DIR= " ${CMAKE_SOURCE_DIR})
message("TestAlg CMAKE_PLATFORM " ${CMAKE_PLATFORM})

# 启用CUDA语言支持
enable_language(CUDA)

# 设置CUDA标准
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fvisibility=hidden -fPIC -shared -fext-numeric-literals")

# C++编译选项（不适用于CUDA）
# 注意：Jetson Orin 使用 CUDA 12.2 和 sm_87 架构
find_package(CUDA 12 REQUIRED)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_87,code=sm_87")
# 添加CUDA编译器标志来解决GCC兼容性问题（Orin 上通常使用系统 GCC）
# set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -ccbin=/usr/bin/gcc-9")
include_directories(${CUDA_INCLUDE_DIRS})
link_directories(${CUDA_TOOLKIT_ROOT_DIR}/lib64)

# TensorRT 8.6.2.3 配置 (Jetson Orin 系统安装)
set(TENSORRT_VERSION "8.6.2.3")
# Orin 上 TensorRT 通过系统包管理器安装，路径固定
set(TENSORRT_INCLUDE_DIR "/usr/include/aarch64-linux-gnu")
set(TENSORRT_LIB_DIR "/usr/lib/aarch64-linux-gnu")

# 验证 TensorRT 头文件和库文件是否存在
if(NOT EXISTS "${TENSORRT_INCLUDE_DIR}/NvInfer.h")
    message(FATAL_ERROR "TensorRT headers not found at ${TENSORRT_INCLUDE_DIR}")
endif()
if(NOT EXISTS "${TENSORRT_LIB_DIR}/libnvinfer.so")
    message(FATAL_ERROR "TensorRT libraries not found at ${TENSORRT_LIB_DIR}")
endif()

# 包含目录
include_directories(
    ${CMAKE_SOURCE_DIR}/Include/AlgParam
    ${CMAKE_SOURCE_DIR}/Include/Common
    ${CMAKE_SOURCE_DIR}/Include/Common/Core
    ${CMAKE_SOURCE_DIR}/Include/Common/Utils
    # ${CMAKE_SOURCE_DIR}/Include/Common/ErrorHandler
    ${CMAKE_SOURCE_DIR}/Include/Common/Factory
    ${CMAKE_SOURCE_DIR}/Include/Common/Interface
    ${CMAKE_SOURCE_DIR}/Include/Common/TensorRT
    ${CMAKE_SOURCE_DIR}/Submodules/ThirdParty/aarch64/opencv/include  # OpenCV 头文件路径
    ${CMAKE_SOURCE_DIR}/Submodules/ThirdParty/aarch64/glog/include  # glog 头文件路径
    ${CMAKE_SOURCE_DIR}/Submodules/ThirdParty/aarch64/fastdds/include
    ${CMAKE_SOURCE_DIR}/Submodules/ThirdParty/aarch64/eigen/include/eigen3  # Eigen 头文件路径（代码中使用 #include <Eigen/Dense>）
    ${CMAKE_SOURCE_DIR}/Src/Common
    ${CMAKE_SOURCE_DIR}/Src/Common/Core
    # ${CMAKE_SOURCE_DIR}/Src/Common/ErrorHandler
    ${CMAKE_SOURCE_DIR}/Src/Common/Factory
    ${CMAKE_SOURCE_DIR}/Src/Common/TensorRT
    ${CMAKE_SOURCE_DIR}/Src/SparseBEV8.6
    ${CMAKE_SOURCE_DIR}/Src/SparseBEV8.6/data
    ${CMAKE_SOURCE_DIR}/Src/SparseBEV8.6/Preprocess
    ${CMAKE_SOURCE_DIR}/Src/SparseBEV8.6/Postprocess
    ${CMAKE_SOURCE_DIR}/Src/SparseBEV8.6/InferenceGPU
    ${TENSORRT_INCLUDE_DIR}
)

# 源文件 - 使用SparseBEV8.6目录
aux_source_directory(${CMAKE_SOURCE_DIR}/Src/SparseBEV8.6 HW_SRC_FILES)
aux_source_directory(${CMAKE_SOURCE_DIR}/Src/SparseBEV8.6/data HW_SRC_FILES)
aux_source_directory(${CMAKE_SOURCE_DIR}/Src/SparseBEV8.6/Preprocess HW_SRC_FILES)
aux_source_directory(${CMAKE_SOURCE_DIR}/Src/SparseBEV8.6/Postprocess HW_SRC_FILES)
aux_source_directory(${CMAKE_SOURCE_DIR}/Src/SparseBEV8.6/InferenceGPU HW_SRC_FILES)
aux_source_directory(. HW_SRC_FILES)

# 添加TensorRT源文件
file(GLOB TENSORRT_SOURCES 
    "${CMAKE_SOURCE_DIR}/Src/Common/TensorRT/*.cpp"
)

# 添加CUDA源文件 - 使用SparseBEV8.6目录
file(GLOB CUDA_SOURCES 
    "${CMAKE_SOURCE_DIR}/Src/SparseBEV8.6/Preprocess/*.cu"
    "${CMAKE_SOURCE_DIR}/Src/SparseBEV8.6/Postprocess/*.cu"
    "${CMAKE_SOURCE_DIR}/Src/SparseBEV8.6/InferenceGPU/*.cu"
)

# 添加RawInferenceResult源文件
list(APPEND HW_SRC_FILES "${CMAKE_SOURCE_DIR}/Src/SparseBEV8.6/data/RawInferenceResult.cpp")

# 链接库路径
link_directories(
    ${CMAKE_SOURCE_DIR}/Output/Lib
    ${CMAKE_SOURCE_DIR}/Submodules/ThirdParty/aarch64/fastdds/lib
    ${CMAKE_SOURCE_DIR}/Submodules/ThirdParty/aarch64/protobuf/lib
    ${CMAKE_SOURCE_DIR}/Submodules/ThirdParty/aarch64/opencv/lib  # OpenCV 库文件路径
    ${CMAKE_SOURCE_DIR}/Submodules/ThirdParty/aarch64/glog/lib
    ${TENSORRT_LIB_DIR}
    ${CUDA_TOOLKIT_ROOT_DIR}/lib64
)

# 生成库 - 使用新的库名
add_library(${PROJECT_NAME} SHARED ${HW_SRC_FILES} ${CUDA_SOURCES} ${TENSORRT_SOURCES} ExportSparseBEVAlgLib.cpp)

# 设置CUDA编译选项
set_target_properties(${PROJECT_NAME} PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES 87  # Jetson Orin 使用 sm_87
    POSITION_INDEPENDENT_CODE ON
)

# 链接依赖库 - 支持TensorRT 8.6
target_link_libraries(${PROJECT_NAME} PRIVATE
    # Common库
    Common
    
    # OpenCV库
    opencv_core
    opencv_imgproc
    opencv_imgcodecs
    opencv_highgui
    opencv_calib3d
    opencv_dnn
    opencv_features2d
    opencv_flann
    opencv_photo
    opencv_stitching
    opencv_video
    opencv_videoio
    opencv_ml
    opencv_objdetect
    
    # 其他依赖
    tinyxml2
    FastddsSer
    ProtoSer
    stdc++fs
    glog
    protobuf
    
    # CUDA
    ${CUDA_LIBRARIES}
    cudart
    cublas
    curand
    
    # TensorRT 8.6
    nvinfer
    nvinfer_plugin
    nvonnxparser
    nvparsers
)