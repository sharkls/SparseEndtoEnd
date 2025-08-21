# Common模块的CMakeLists.txt
project(Common)

# 包含目录
include_directories(
    ${CMAKE_SOURCE_DIR}/Include
    ${CMAKE_SOURCE_DIR}/Include/Common
    ${CMAKE_SOURCE_DIR}/Include/Common/Core
    ${CMAKE_SOURCE_DIR}/Include/Common/Factory
    ${CMAKE_SOURCE_DIR}/Include/Common/Interface
    ${CMAKE_SOURCE_DIR}/Include/Common/Utils
    ${CMAKE_SOURCE_DIR}/Include/Common/TensorRT
)

# 源文件
aux_source_directory(ModuleFactory COMMON_SRC_FILES)
aux_source_directory(Core COMMON_SRC_FILES)

# 条件编译TensorRT相关文件
if(BUILD_SPARSEBEV OR BUILD_SPARSEBEV8_6)
    aux_source_directory(TensorRT COMMON_SRC_FILES)
    
    
    
    # 添加TensorRT头文件路径
    if(BUILD_SPARSEBEV)
        # 添加CUDA头文件路径（TensorRT需要）
        find_package(CUDA 12 REQUIRED)
        include_directories(${CUDA_INCLUDE_DIRS})
        # TensorRT 10.2
        set(TENSORRT_ROOT "/mnt/env/TensorRT-10.2.0.19")
        include_directories(${TENSORRT_ROOT}/include)
    elseif(BUILD_SPARSEBEV8_6)
        # 添加CUDA头文件路径（TensorRT需要）
        find_package(CUDA 11 REQUIRED)
        include_directories(${CUDA_INCLUDE_DIRS})
        # TensorRT 8.6
        set(TENSORRT_ROOT "/mnt/env/tensorrt/TensorRT-8.5.1.7")
        include_directories(${TENSORRT_ROOT}/include)
    endif()
    
    # 添加TensorRT库路径
    link_directories(${TENSORRT_ROOT}/lib)
endif()

# 生成静态库
add_library(Common STATIC ${COMMON_SRC_FILES})

# 设置编译选项
set_target_properties(Common PROPERTIES
    POSITION_INDEPENDENT_CODE ON
)

# 链接依赖库
target_link_libraries(Common PRIVATE
    glog
    protobuf
    tinyxml2
)

# 条件链接TensorRT库
if(BUILD_SPARSEBEV OR BUILD_SPARSEBEV8_6)
    target_link_libraries(Common PRIVATE
        nvinfer
        nvinfer_plugin
        nvonnxparser
    )
endif() 