# SparseBEV8.6 特征提取模块单元测试 CMakeLists.txt

# 使用项目中的GoogleTest
set(GTEST_ROOT "${CMAKE_SOURCE_DIR}/Submodules/ThirdParty/googletest")
set(GTEST_INCLUDE_DIR "${GTEST_ROOT}/googletest/include")
set(GTEST_LIB_DIR "${GTEST_ROOT}/build/lib")

# 设置测试可执行文件
set(TEST_EXECUTABLE_NAME "sparse4d_extract_feat_unit_test")

# 添加测试源文件
set(TEST_SOURCES
    sparse4d_extract_feat_unit_test.cpp
)

# 创建测试可执行文件
add_executable(${TEST_EXECUTABLE_NAME} ${TEST_SOURCES})

# 设置编译选项
target_compile_features(${TEST_EXECUTABLE_NAME} PRIVATE cxx_std_14)

# 设置包含目录
target_include_directories(${TEST_EXECUTABLE_NAME} PRIVATE
    ${CMAKE_SOURCE_DIR}/Include
    ${CMAKE_SOURCE_DIR}/Include/Common
    ${CMAKE_SOURCE_DIR}/Include/Common/Core
    ${CMAKE_SOURCE_DIR}/Include/Common/Utils
    ${CMAKE_SOURCE_DIR}/Include/Common/Interface
    ${CMAKE_SOURCE_DIR}/Include/Common/Factory
    ${CMAKE_SOURCE_DIR}/Include/Common/TensorRT
    ${CMAKE_SOURCE_DIR}/Src/Common
    ${CMAKE_SOURCE_DIR}/Src/Common/Core
    ${CMAKE_SOURCE_DIR}/Src/Common/Factory
    ${CMAKE_SOURCE_DIR}/Src/Common/TensorRT
    ${CMAKE_SOURCE_DIR}/Src/SparseBEV8.6
    ${CMAKE_SOURCE_DIR}/Src/SparseBEV8.6/Inference
    ${CMAKE_SOURCE_DIR}/Src/SparseBEV8.6/data
    ${GTEST_INCLUDE_DIR}
    ${TENSORRT_INCLUDE_DIR}
    ${CUDA_INCLUDE_DIRS}
)

# 链接库
target_link_libraries(${TEST_EXECUTABLE_NAME} PRIVATE
    # GoogleTest库
    ${GTEST_LIB_DIR}/libgtest.a
    ${GTEST_LIB_DIR}/libgtest_main.a
    ${GTEST_LIB_DIR}/libgmock.a
    ${GTEST_LIB_DIR}/libgmock_main.a
    
    # 项目库
    SparseBEVAlg8.6
    Common
    
    # CUDA
    ${CUDA_LIBRARIES}
    cudart
    cublas
    curand
    
    # TensorRT
    nvinfer
    nvinfer_plugin
    nvonnxparser
    nvparsers
    
    # 其他依赖
    glog
    protobuf
    stdc++fs
    pthread
)

# 设置CUDA编译选项
set_target_properties(${TEST_EXECUTABLE_NAME} PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES 86
)

# 设置CUDA标准（如果支持）
if(CMAKE_CUDA_COMPILER)
    set_target_properties(${TEST_EXECUTABLE_NAME} PROPERTIES
        CUDA_STANDARD 14
        CUDA_STANDARD_REQUIRED ON
    )
endif()

# 添加编译定义
target_compile_definitions(${TEST_EXECUTABLE_NAME} PRIVATE
    TENSORRT_VERSION_8_6
    TENSORRT_MAJOR=8
    TENSORRT_MINOR=6
    GTEST_HAS_PTHREAD=1
)

# 设置输出目录
set_target_properties(${TEST_EXECUTABLE_NAME} PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/Output/unit_test
)

# 添加测试
add_test(NAME ${TEST_EXECUTABLE_NAME} COMMAND ${TEST_EXECUTABLE_NAME})

# 设置测试环境变量
set_tests_properties(${TEST_EXECUTABLE_NAME} PROPERTIES
    ENVIRONMENT "LD_LIBRARY_PATH=${CMAKE_SOURCE_DIR}/Output/Lib:${TENSORRT_LIB_DIR}:${CUDA_TOOLKIT_ROOT_DIR}/lib64:$ENV{LD_LIBRARY_PATH}"
)

# 打印构建信息
message(STATUS "Building SparseBEV8.6 feature extraction unit test: ${TEST_EXECUTABLE_NAME}")
message(STATUS "Test executable will be placed in: ${CMAKE_SOURCE_DIR}/Output/unit_test")
message(STATUS "Using GoogleTest from: ${GTEST_ROOT}") 