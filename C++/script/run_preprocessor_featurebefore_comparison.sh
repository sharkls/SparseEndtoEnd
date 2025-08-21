#!/bin/bash

# 特征提取结果对比脚本

# 设置路径
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
COMPARISON_SCRIPT="$SCRIPT_DIR/compare_feature_bins.py"

# 检查分析脚本是否存在
if [ ! -f "$COMPARISON_SCRIPT" ]; then
    echo "[ERROR] 分析脚本不存在: $COMPARISON_SCRIPT"
    exit 1
fi

# 设置默认参数
CURRENT_FILE="/share/Code/SparseEnd2End/C++/Output/val_bin/image_1*6*3*256*704_float32.bin"
EXPECTED_FILE="/share/Code/SparseEnd2End/script/tutorial/asset/sample_0_imgs_1*6*3*256*704_float32.bin"
TOLERANCE=1e-2

# 函数：显示帮助信息
show_help() {
    echo "用法: $0 [选项]"
    echo ""
    echo "选项:"
    echo "  -c, --current FILE     当前特征提取结果文件路径"
    echo "  -e, --expected FILE    期望的特征提取结果文件路径"
    echo "  -t, --tolerance F      误差容差 (默认: $TOLERANCE)"
    echo "  -h, --help             显示此帮助信息"
    echo ""
    echo "示例:"
    echo "  $0 --current /path/to/current.bin --expected /path/to/expected.bin"
    echo "  $0 -c /path/to/current.bin -e /path/to/expected.bin -t 1e-5"
}

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        -c|--current)
            CURRENT_FILE="$2"
            shift 2
            ;;
        -e|--expected)
            EXPECTED_FILE="$2"
            shift 2
            ;;
        -t|--tolerance)
            TOLERANCE="$2"
            shift 2
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            echo "[ERROR] 未知选项: $1"
            show_help
            exit 1
            ;;
    esac
done

# 检查文件是否存在
if [ ! -f "$CURRENT_FILE" ]; then
    echo "[ERROR] 当前文件不存在: $CURRENT_FILE"
    exit 1
fi

if [ ! -f "$EXPECTED_FILE" ]; then
    echo "[ERROR] 期望文件不存在: $EXPECTED_FILE"
    exit 1
fi

# 运行对比分析
echo "[INFO] 开始特征提取结果对比分析..."
echo "[INFO] 当前文件: $CURRENT_FILE"
echo "[INFO] 期望文件: $EXPECTED_FILE"
echo "[INFO] 误差容差: $TOLERANCE"
echo ""

python3 "$COMPARISON_SCRIPT" \
    --current "$CURRENT_FILE" \
    --expected "$EXPECTED_FILE" \
    --tolerance "$TOLERANCE"

# 检查脚本执行结果
if [ $? -eq 0 ]; then
    echo ""
    echo "[INFO] 特征提取结果对比分析完成"
else
    echo ""
    echo "[ERROR] 特征提取结果对比分析失败"
    exit 1
fi 