project(SparseBEVAlg)

message("CMAKE_SOURCE_DIR= " ${CMAKE_SOURCE_DIR})
message("TestAlg CMAKE_PLATFORM " ${CMAKE_PLATFORM})

# 启用CUDA语言支持
enable_language(CUDA)

# 编译选项
add_compile_options(
    -fvisibility=hidden
    -fPIC
    -shared
    -fext-numeric-literals
)

# CUDA编译选项
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_86,code=sm_86")
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# CUDA配置
find_package(CUDA 12 REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})

# TensorRT配置 - TensorRT 10.2
set(TENSORRT_VERSION "10.2")
set(TENSORRT_ROOT "/mnt/env/TensorRT-${TENSORRT_VERSION}.0.19")
if(NOT EXISTS ${TENSORRT_ROOT})
    message(FATAL_ERROR "TensorRT ${TENSORRT_VERSION} not found at ${TENSORRT_ROOT}")
endif()

include_directories(${TENSORRT_ROOT}/include)
link_directories(${TENSORRT_ROOT}/lib)

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
    ${CMAKE_SOURCE_DIR}/Src/SparseBEV
    ${CMAKE_SOURCE_DIR}/Src/SparseBEV/Preprocess
    ${CMAKE_SOURCE_DIR}/Src/SparseBEV/Postprocess
    ${CMAKE_SOURCE_DIR}/Src/SparseBEV/Inference
)

# 源文件
# aux_source_directory(${CMAKE_SOURCE_DIR}/Src/Common/Core HW_SRC_FILES)
# aux_source_directory(${CMAKE_SOURCE_DIR}/Src/Common/Factory HW_SRC_FILES)
# aux_source_directory(${CMAKE_SOURCE_DIR}/Src/Common/TensorRT HW_SRC_FILES)
aux_source_directory(${CMAKE_SOURCE_DIR}/Src/SparseBEV HW_SRC_FILES)
aux_source_directory(${CMAKE_SOURCE_DIR}/Src/SparseBEV/Preprocess HW_SRC_FILES)
aux_source_directory(${CMAKE_SOURCE_DIR}/Src/SparseBEV/Postprocess HW_SRC_FILES)
aux_source_directory(${CMAKE_SOURCE_DIR}/Src/SparseBEV/Inference HW_SRC_FILES)
aux_source_directory(. HW_SRC_FILES)

# 添加TensorRT源文件
file(GLOB TENSORRT_SOURCES 
    "${CMAKE_SOURCE_DIR}/Src/Common/TensorRT/*.cpp"
)

# 添加CUDA源文件
file(GLOB CUDA_SOURCES 
    "${CMAKE_SOURCE_DIR}/Src/SparseBEV/Preprocess/*.cu"
    "${CMAKE_SOURCE_DIR}/Src/SparseBEV/Postprocess/*.cu"
)

# 添加RawInferenceResult源文件
list(APPEND HW_SRC_FILES "${CMAKE_SOURCE_DIR}/Src/SparseBEV/Inference/RawInferenceResult.cpp")

# 链接库路径
link_directories(
    ${CMAKE_SOURCE_DIR}/Output/Lib
    ${CMAKE_SOURCE_DIR}/Submodules/TPL
    ${CMAKE_SOURCE_DIR}/Submodules/TPL/fastdds
    ${CMAKE_SOURCE_DIR}/Submodules/TPL/protobuf
    ${CMAKE_SOURCE_DIR}/Submodules/TPL/av_opencv
    ${CMAKE_SOURCE_DIR}/Submodules/TPL/glog
    ${CUDA_TOOLKIT_ROOT_DIR}/lib64
)

# 生成库
add_library(${PROJECT_NAME} SHARED ${HW_SRC_FILES} ${CUDA_SOURCES} ${TENSORRT_SOURCES} ExportSparseBEVAlgLib.cpp)

# 设置CUDA编译选项
set_target_properties(${PROJECT_NAME} PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES 86
    POSITION_INDEPENDENT_CODE ON
)

# 链接依赖库
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
    
    # TensorRT
    nvinfer
    nvinfer_plugin
    nvonnxparser
)