# GPU NMS单元测试CMakeLists.txt

# 启用CUDA语言支持
enable_language(CUDA)

# 设置CUDA编译选项
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_86,code=sm_86")
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# 包含目录
include_directories(
    ${CMAKE_SOURCE_DIR}/Include/Common
    ${CMAKE_SOURCE_DIR}/Src/SparseBEV/Postprocess
    ${CUDA_INCLUDE_DIRS}
)

# 源文件
set(GPU_NMS_TEST_SOURCES
    gpu_nms_unit_test.cpp
    ../gpu_nms.cu
    ../gpu_nms_wrapper.cpp
)

# 创建测试可执行文件
add_executable(gpu_nms_unit_test ${GPU_NMS_TEST_SOURCES})

# 设置CUDA编译选项
set_target_properties(gpu_nms_unit_test PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES 86
)

# 链接依赖库
target_link_libraries(gpu_nms_unit_test PRIVATE
    # CUDA
    ${CUDA_LIBRARIES}
    cudart
    cublas
    curand
    thrust
    
    # 其他依赖
    stdc++
)

# 添加测试
add_test(NAME GPU_NMS_Unit_Test COMMAND gpu_nms_unit_test)

# 设置测试工作目录
set_tests_properties(GPU_NMS_Unit_Test PROPERTIES
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
) 