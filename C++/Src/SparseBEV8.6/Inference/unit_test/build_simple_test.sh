#!/bin/bash

# 简单单元测试构建脚本（不依赖CUDA）

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

print_info "开始构建简单单元测试..."

# 创建临时构建目录
BUILD_DIR="$CPP_DIR/build_simple_test"
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

print_info "构建目录: $BUILD_DIR"

# 复制简单的CMakeLists.txt
cp "$SCRIPT_DIR/CMakeLists_simple.txt" "$SCRIPT_DIR/CMakeLists.txt.bak"
cp "$SCRIPT_DIR/CMakeLists_simple.txt" "$SCRIPT_DIR/CMakeLists.txt"

# 配置CMake
print_info "配置CMake..."
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_TESTING=ON

if [ $? -eq 0 ]; then
    print_success "CMake配置成功"
else
    print_error "CMake配置失败"
    # 恢复原始CMakeLists.txt
    mv "$SCRIPT_DIR/CMakeLists.txt.bak" "$SCRIPT_DIR/CMakeLists.txt"
    exit 1
fi

# 尝试编译
print_info "尝试编译..."
make simple_unit_test -j$(nproc)

if [ $? -eq 0 ]; then
    print_success "编译成功！"
    
    # 检查生成的可执行文件
    TEST_EXECUTABLE="$CPP_DIR/Output/unit_test/simple_unit_test"
    if [ -f "$TEST_EXECUTABLE" ]; then
        print_success "测试可执行文件已生成: $TEST_EXECUTABLE"
        ls -la "$TEST_EXECUTABLE"
        
        # 运行测试
        print_info "运行简单测试..."
        "$TEST_EXECUTABLE"
        
        if [ $? -eq 0 ]; then
            print_success "简单测试通过！"
        else
            print_error "简单测试失败"
        fi
    else
        print_error "测试可执行文件未找到"
    fi
else
    print_error "编译失败"
fi

# 恢复原始CMakeLists.txt
mv "$SCRIPT_DIR/CMakeLists.txt.bak" "$SCRIPT_DIR/CMakeLists.txt"

# 清理
print_info "清理临时构建目录..."
cd "$PROJECT_ROOT"
rm -rf "$BUILD_DIR"

print_success "简单测试构建完成！" 