#!/bin/bash

# SparseBEV8.6 特征提取模块单元测试构建和运行脚本

set -e  # 遇到错误时退出

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 打印带颜色的消息
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 获取脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../../../.." && pwd)"
CPP_DIR="$PROJECT_ROOT/C++"

print_info "脚本目录: $SCRIPT_DIR"
print_info "项目根目录: $PROJECT_ROOT"

# 检查CUDA环境
check_cuda() {
    print_info "检查CUDA环境..."
    
    if ! command -v nvcc &> /dev/null; then
        print_error "CUDA编译器(nvcc)未找到"
        return 1
    fi
    
    if ! command -v nvidia-smi &> /dev/null; then
        print_error "nvidia-smi未找到"
        return 1
    fi
    
    print_success "CUDA环境检查通过"
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader,nounits
}

# 检查TensorRT环境
check_tensorrt() {
    print_info "检查TensorRT环境..."
    
    TENSORRT_ROOT="/mnt/env/tensorrt/TensorRT-8.5.1.7"
    if [ ! -d "$TENSORRT_ROOT" ]; then
        print_warning "TensorRT目录不存在: $TENSORRT_ROOT"
        print_warning "请确保TensorRT已正确安装"
        return 1
    fi
    
    if [ ! -f "$TENSORRT_ROOT/lib/libnvinfer.so" ]; then
        print_error "TensorRT库文件未找到"
        return 1
    fi
    
    print_success "TensorRT环境检查通过"
}

# 检查GoogleTest
check_gtest() {
    print_info "检查GoogleTest..."
    
    GTEST_ROOT="/share/Code/SparseEnd2End/C++/Submodules/ThirdParty/googletest"
    GTEST_LIB_DIR="$GTEST_ROOT/build/lib"
    
    if [ ! -d "$GTEST_ROOT" ]; then
        print_error "GoogleTest目录不存在: $GTEST_ROOT"
        return 1
    fi
    
    if [ ! -f "$GTEST_ROOT/googletest/include/gtest/gtest.h" ]; then
        print_error "GoogleTest头文件未找到"
        return 1
    fi
    
    if [ ! -f "$GTEST_LIB_DIR/libgtest.a" ]; then
        print_warning "GoogleTest库文件未找到，可能需要构建GoogleTest"
        print_info "请运行: cd $GTEST_ROOT && mkdir -p build && cd build && cmake .. && make"
        return 1
    fi
    
    print_success "GoogleTest检查通过"
    print_info "GoogleTest路径: $GTEST_ROOT"
    print_info "GoogleTest库路径: $GTEST_LIB_DIR"
}

# 创建构建目录
create_build_dir() {
    print_info "创建构建目录..."
    
    BUILD_DIR="$CPP_DIR/build"
    mkdir -p "$BUILD_DIR"
    cd "$BUILD_DIR"
    
    print_success "构建目录: $BUILD_DIR"
}

# 配置CMake
configure_cmake() {
    print_info "配置CMake..."
    
    cmake .. \
        -DCMAKE_BUILD_TYPE=Release \
        -DBUILD_TESTING=ON \
        -DCMAKE_CUDA_ARCHITECTURES=86
    
    if [ $? -eq 0 ]; then
        print_success "CMake配置成功"
    else
        print_error "CMake配置失败"
        exit 1
    fi
}

# 构建项目
build_project() {
    print_info "构建项目..."
    
    # 获取CPU核心数
    CPU_CORES=$(nproc)
    print_info "使用 $CPU_CORES 个CPU核心进行构建"
    
    make -j$CPU_CORES
    
    if [ $? -eq 0 ]; then
        print_success "项目构建成功"
    else
        print_error "项目构建失败"
        exit 1
    fi
}

# 运行测试
run_tests() {
    print_info "运行单元测试..."
    
    TEST_EXECUTABLE="$PROJECT_ROOT/Output/unit_test/sparse4d_extract_feat_unit_test"
    
    if [ ! -f "$TEST_EXECUTABLE" ]; then
        print_error "测试可执行文件未找到: $TEST_EXECUTABLE"
        exit 1
    fi
    
    # 设置环境变量
    export LD_LIBRARY_PATH="$PROJECT_ROOT/Output/Lib:/mnt/env/tensorrt/TensorRT-8.5.1.7/lib:$LD_LIBRARY_PATH"
    
    print_info "运行测试: $TEST_EXECUTABLE"
    print_info "环境变量: LD_LIBRARY_PATH=$LD_LIBRARY_PATH"
    
    # 运行测试
    "$TEST_EXECUTABLE"
    
    if [ $? -eq 0 ]; then
        print_success "所有测试通过"
    else
        print_error "测试失败"
        exit 1
    fi
}

# 清理构建文件
clean_build() {
    print_info "清理构建文件..."
    
    if [ -d "$PROJECT_ROOT/build" ]; then
        rm -rf "$PROJECT_ROOT/build"
        print_success "构建文件已清理"
    fi
}

# 显示帮助信息
show_help() {
    echo "SparseBEV8.6 特征提取模块单元测试构建和运行脚本"
    echo ""
    echo "用法: $0 [选项]"
    echo ""
    echo "选项:"
    echo "  -h, --help     显示此帮助信息"
    echo "  -c, --clean    清理构建文件"
    echo "  -e, --env      仅检查环境"
    echo "  -b, --build    仅构建项目"
    echo "  -t, --test     仅运行测试"
    echo "  -a, --all      完整流程（检查环境 -> 构建 -> 测试）"
    echo ""
    echo "示例:"
    echo "  $0 -a          # 完整流程"
    echo "  $0 -e          # 仅检查环境"
    echo "  $0 -b          # 仅构建"
    echo "  $0 -t          # 仅运行测试"
    echo "  $0 -c          # 清理构建文件"
}

# 主函数
main() {
    print_info "=========================================="
    print_info "SparseBEV8.6 特征提取模块单元测试"
    print_info "=========================================="
    
    # 解析命令行参数
    case "${1:-}" in
        -h|--help)
            show_help
            exit 0
            ;;
        -c|--clean)
            clean_build
            exit 0
            ;;
        -e|--env)
            check_cuda
            check_tensorrt
            check_gtest
            exit 0
            ;;
        -b|--build)
            create_build_dir
            configure_cmake
            build_project
            exit 0
            ;;
        -t|--test)
            run_tests
            exit 0
            ;;
        -a|--all|"")
            # 完整流程
            check_cuda
            check_tensorrt
            check_gtest
            create_build_dir
            configure_cmake
            build_project
            run_tests
            ;;
        *)
            print_error "未知选项: $1"
            show_help
            exit 1
            ;;
    esac
}

# 运行主函数
main "$@" 