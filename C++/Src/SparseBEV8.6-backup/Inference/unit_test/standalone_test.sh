#!/bin/bash

# 完全独立的单元测试脚本

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 获取脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../../../.." && pwd)"
CPP_DIR="$PROJECT_ROOT/C++"

print_info "开始构建独立单元测试..."

# 创建独立的构建目录
BUILD_DIR="$SCRIPT_DIR/build_standalone"
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

print_info "构建目录: $BUILD_DIR"

# 创建独立的CMakeLists.txt
cat > CMakeLists.txt << 'EOF'
cmake_minimum_required(VERSION 3.16)
project(StandaloneUnitTest)

# 设置C++标准
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 使用项目中的GoogleTest
set(GTEST_ROOT "/share/Code/SparseEnd2End/C++/Submodules/ThirdParty/googletest")
set(GTEST_INCLUDE_DIR "${GTEST_ROOT}/googletest/include")
set(GTEST_LIB_DIR "${GTEST_ROOT}/build/lib")

# 设置测试可执行文件
set(TEST_EXECUTABLE_NAME "standalone_unit_test")

# 添加测试源文件
set(TEST_SOURCES
    ../simple_test.cpp
)

# 创建测试可执行文件
add_executable(${TEST_EXECUTABLE_NAME} ${TEST_SOURCES})

# 设置包含目录
target_include_directories(${TEST_EXECUTABLE_NAME} PRIVATE
    ${GTEST_INCLUDE_DIR}
)

# 链接库
target_link_libraries(${TEST_EXECUTABLE_NAME} PRIVATE
    # GoogleTest库
    ${GTEST_LIB_DIR}/libgtest.a
    ${GTEST_LIB_DIR}/libgtest_main.a
    ${GTEST_LIB_DIR}/libgmock.a
    ${GTEST_LIB_DIR}/libgmock_main.a
    
    # 系统库
    pthread
)

# 添加编译定义
target_compile_definitions(${TEST_EXECUTABLE_NAME} PRIVATE
    GTEST_HAS_PTHREAD=1
)

# 打印构建信息
message(STATUS "Building standalone unit test: ${TEST_EXECUTABLE_NAME}")
message(STATUS "Using GoogleTest from: ${GTEST_ROOT}")
EOF

# 配置CMake
print_info "配置CMake..."
cmake .

if [ $? -eq 0 ]; then
    print_success "CMake配置成功"
else
    print_error "CMake配置失败"
    exit 1
fi

# 尝试编译
print_info "尝试编译..."
make -j$(nproc)

if [ $? -eq 0 ]; then
    print_success "编译成功！"
    
    # 检查生成的可执行文件
    TEST_EXECUTABLE="$BUILD_DIR/standalone_unit_test"
    if [ -f "$TEST_EXECUTABLE" ]; then
        print_success "测试可执行文件已生成: $TEST_EXECUTABLE"
        ls -la "$TEST_EXECUTABLE"
        
        # 运行测试
        print_info "运行独立测试..."
        "$TEST_EXECUTABLE"
        
        if [ $? -eq 0 ]; then
            print_success "独立测试通过！"
        else
            print_error "独立测试失败"
        fi
    else
        print_error "测试可执行文件未找到"
    fi
else
    print_error "编译失败"
fi

# 清理
print_info "清理构建目录..."
cd "$SCRIPT_DIR"
rm -rf "$BUILD_DIR"

print_success "独立测试构建完成！" 