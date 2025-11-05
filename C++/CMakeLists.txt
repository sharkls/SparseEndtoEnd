# 需要的CMake的最小版本号
cmake_minimum_required(VERSION 3.10)

# 执行文件另取别名
project(testAlgLib)

# 编译设置
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_PLATFORM 86)

# 编译选项
option(BUILD_SPARSEBEV "Build SparseBEV with TensorRT 10.2" OFF)
option(BUILD_SPARSEBEV8_6 "Build SparseBEV8.6 with TensorRT 8.6" ON)

add_compile_options(-shared)
add_definitions(-w)   # 不显示warning

message("CMAKE_SOURCE_DIR= " ${CMAKE_SOURCE_DIR})
message("BUILD_SPARSEBEV: " ${BUILD_SPARSEBEV})
message("BUILD_SPARSEBEV8_6: " ${BUILD_SPARSEBEV8_6})

# 输出路径
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/Output/Lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/Output/Lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/Output)

# 包含头文件的路径(Fastdds 、 Proto定义文件)
include_directories(${CMAKE_SOURCE_DIR}/Submodules/FastddsSer/data)
include_directories(${CMAKE_SOURCE_DIR}/Submodules/ProtoSer/param)

# 包含头文件的路径
include_directories(${CMAKE_SOURCE_DIR}/Include)
include_directories(${CMAKE_SOURCE_DIR}/Include/Common/Core)
include_directories(${CMAKE_SOURCE_DIR}/Include/Common/Utils)
include_directories(${CMAKE_SOURCE_DIR}/Include/Common/ErrorHandler)
include_directories(${CMAKE_SOURCE_DIR}/Include/Common/Factory)
include_directories(${CMAKE_SOURCE_DIR}/Include/Common/Interface)
include_directories(${CMAKE_SOURCE_DIR}/Include/Interface)
include_directories(${CMAKE_SOURCE_DIR}/Include/Common/TensorRT)
include_directories(${CMAKE_SOURCE_DIR}/Src)

# 优先使用项目自带的第三方库，避免系统库冲突
include_directories(${CMAKE_SOURCE_DIR}/Submodules/ThirdParty/protobuf/include)
include_directories(${CMAKE_SOURCE_DIR}/Submodules/ThirdParty/fastdds/include)
include_directories(${CMAKE_SOURCE_DIR}/Submodules/ThirdParty/av_opencv/include)
include_directories(${CMAKE_SOURCE_DIR}/Submodules/ThirdParty/tinyxml2/include)
include_directories(${CMAKE_SOURCE_DIR}/Submodules/ThirdParty/googletest/include)
include_directories(${CMAKE_SOURCE_DIR}/Submodules/ThirdParty/gflags/include)
include_directories(${CMAKE_SOURCE_DIR}/Submodules/ThirdParty/eigen/include)
include_directories(${CMAKE_SOURCE_DIR}/Submodules/ThirdParty/glog/include)

# 链接动态库的路径
link_directories(${CMAKE_SOURCE_DIR}/Output/Lib)
link_directories(${CMAKE_SOURCE_DIR}/Submodules/ThirdParty/protobuf/lib)
link_directories(${CMAKE_SOURCE_DIR}/Submodules/ThirdParty/fastdds/lib)
link_directories(${CMAKE_SOURCE_DIR}/Submodules/ThirdParty/av_opencv/lib)
link_directories(${CMAKE_SOURCE_DIR}/Submodules/ThirdParty/googletest/lib)
link_directories(${CMAKE_SOURCE_DIR}/Submodules/ThirdParty/gflags/lib)
link_directories(${CMAKE_SOURCE_DIR}/Submodules/ThirdParty/tinyxml2/lib)
link_directories(${CMAKE_SOURCE_DIR}/Submodules/ThirdParty/glog/lib)

# 子目录的路径
add_subdirectory(Submodules/FastddsSer)
add_subdirectory(Submodules/ProtoSer)
add_subdirectory(Src)

# # 条件添加SparseBEV子目录
# if(BUILD_SPARSEBEV)
#     add_subdirectory(Src/SparseBEV)
# endif()

# if(BUILD_SPARSEBEV8_6)
#     add_subdirectory(Src/SparseBEV8.6)
#     # add_subdirectory(Src/SparseBEVFP16)
# endif()

# # 添加Common模块（总是需要）
# add_subdirectory(Src/Common)

# 源文件的路径
aux_source_directory(. HW_SRC_FILES)

if(CMAKE_PLATFORM MATCHES "86")
    #生成可执行程序
    add_executable(${PROJECT_NAME} main.cpp
                                # TestSparseBEVAlg.cpp
                                # TestSparseBEVAlgv2.cpp
                                TestSparseBEV8.6.cpp
                                TestSparse4D.cpp
    )

    # 首先链接protobuf相关库
    target_link_libraries(${PROJECT_NAME} PRIVATE protobuf)
    target_link_libraries(${PROJECT_NAME} PRIVATE ProtoSer)
    target_link_libraries(${PROJECT_NAME} PRIVATE FastddsSer)

    # 然后链接OpenCV库
    target_link_libraries(${PROJECT_NAME} PRIVATE opencv_core)
    target_link_libraries(${PROJECT_NAME} PRIVATE opencv_imgproc)
    target_link_libraries(${PROJECT_NAME} PRIVATE opencv_imgcodecs)
    target_link_libraries(${PROJECT_NAME} PRIVATE opencv_highgui)
    target_link_libraries(${PROJECT_NAME} PRIVATE opencv_calib3d)
    target_link_libraries(${PROJECT_NAME} PRIVATE opencv_dnn)
    target_link_libraries(${PROJECT_NAME} PRIVATE opencv_features2d)
    target_link_libraries(${PROJECT_NAME} PRIVATE opencv_flann)
    target_link_libraries(${PROJECT_NAME} PRIVATE opencv_ml)
    target_link_libraries(${PROJECT_NAME} PRIVATE opencv_objdetect)
    target_link_libraries(${PROJECT_NAME} PRIVATE opencv_photo)
    target_link_libraries(${PROJECT_NAME} PRIVATE opencv_stitching)
    target_link_libraries(${PROJECT_NAME} PRIVATE opencv_video)
    target_link_libraries(${PROJECT_NAME} PRIVATE opencv_videoio)

    # 其他库
    target_link_libraries(${PROJECT_NAME} PRIVATE tinyxml2)
    target_link_libraries(${PROJECT_NAME} PRIVATE stdc++fs)
    target_link_libraries(${PROJECT_NAME} PRIVATE pthread)
    target_link_libraries(${PROJECT_NAME} PRIVATE glog)

    # 最后链接算法模块的动态库（CUDA和TensorRT依赖由子模块管理）
    if(BUILD_SPARSEBEV)
        target_link_libraries(${PROJECT_NAME} PRIVATE SparseBEVAlg)
    endif()
    
    if(BUILD_SPARSEBEV8_6)
        target_link_libraries(${PROJECT_NAME} PRIVATE SparseBEVAlg8.6)
        # Ensure the factory symbol CreateCoreObj is not dropped by --as-needed
        target_link_libraries(${PROJECT_NAME} PRIVATE "-Wl,--no-as-needed" Sparse4DAlg "-Wl,--as-needed")
    endif()

endif()