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
find_package(CUDA 11 REQUIRED)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_86,code=sm_86")
# 添加CUDA编译器标志来解决GCC兼容性问题
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -ccbin=/usr/bin/gcc-9")
include_directories(${CUDA_INCLUDE_DIRS})
link_directories(${CUDA_TOOLKIT_ROOT_DIR}/lib64)

# TensorRT 8.6 配置
set(TENSORRT_VERSION "8.5.1.7")
set(TENSORRT_ROOT "/mnt/env/tensorrt/TensorRT-${TENSORRT_VERSION}")
if(NOT EXISTS ${TENSORRT_ROOT})
    message(FATAL_ERROR "TensorRT ${TENSORRT_VERSION} not found at ${TENSORRT_ROOT}")
endif()

set(TENSORRT_INCLUDE_DIR "${TENSORRT_ROOT}/include")
set(TENSORRT_LIB_DIR "${TENSORRT_ROOT}/lib")

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
    ${CMAKE_SOURCE_DIR}/Include/Thirdpart/av_opencv
    ${CMAKE_SOURCE_DIR}/Include/Thirdpart/glog
    ${CMAKE_SOURCE_DIR}/Include/Thirdpart/fastdds
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
    ${CMAKE_SOURCE_DIR}/Submodules/TPL
    ${CMAKE_SOURCE_DIR}/Submodules/TPL/fastdds
    ${CMAKE_SOURCE_DIR}/Submodules/TPL/protobuf
    ${CMAKE_SOURCE_DIR}/Submodules/TPL/av_opencv
    ${CMAKE_SOURCE_DIR}/Submodules/TPL/glog
    ${TENSORRT_LIB_DIR}
    ${CUDA_TOOLKIT_ROOT_DIR}/lib64
)

# 生成库 - 使用新的库名
add_library(${PROJECT_NAME} SHARED ${HW_SRC_FILES} ${CUDA_SOURCES} ${TENSORRT_SOURCES} ExportSparseBEVAlgLib.cpp)

# 设置CUDA编译选项
set_target_properties(${PROJECT_NAME} PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES 86
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