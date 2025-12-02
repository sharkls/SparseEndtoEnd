#!/bin/bash
# GDB调试脚本：用于调试TensorRT引擎构建时的段错误
# 使用方法: ./debug_build_engine.sh [fp32|fp16|int8] [head1|head2]
# 示例: ./debug_build_engine.sh fp16 head1

# 加载环境设置
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/tools/set_env.sh"

# 解析精度参数
PRECISION=${1:-fp16}

# 根据精度参数生成TensorRT参数
get_precision_args() {
    case "$PRECISION" in
        "fp32")
            echo ""
            ;;
        "fp16")
            echo "--fp16"
            ;;
        "int8")
            echo "--int8 --strictTypeConstraints"
            ;;
    esac
}

PRECISION_ARGS=$(get_precision_args)

# 组合插件参数
PLUGIN_ARGS=""
if [[ -n "${ENVTARGETPLUGIN}" && -f "${ENVTARGETPLUGIN}" ]]; then
    PLUGIN_ARGS="${PLUGIN_ARGS} --plugins=${ENVTARGETPLUGIN}"
    echo "[INFO] DeformableAttentionAggrPlugin enabled: ${ENVTARGETPLUGIN}"
fi
if [[ -n "${ENV_LAYER_NORM_PLUGIN}" && -f "${ENV_LAYER_NORM_PLUGIN}" ]]; then
    PLUGIN_ARGS="${PLUGIN_ARGS} --plugins=${ENV_LAYER_NORM_PLUGIN}"
    echo "[INFO] LayerNormPlugin enabled: ${ENV_LAYER_NORM_PLUGIN}"
fi
if [[ -n "${ENV_SPARSEBOX_PLUGIN}" && -f "${ENV_SPARSEBOX_PLUGIN}" ]]; then
    PLUGIN_ARGS="${PLUGIN_ARGS} --plugins=${ENV_SPARSEBOX_PLUGIN}"
    echo "[INFO] SparseBox3DKeyPointsPlugin enabled: ${ENV_SPARSEBOX_PLUGIN}"
fi

echo "===================================================================================================================="
echo "使用GDB调试TensorRT引擎构建"
echo "精度: $PRECISION"
echo "===================================================================================================================="

# 选择要调试的引擎（head1或head2）
ENGINE_TYPE=${2:-head1}

if [ "$ENGINE_TYPE" == "head1" ]; then
    ONNX_FILE=${ENV_HEAD1_ONNX}
    ENGINE_FILE=${ENV_HEAD1_ENGINE}
    LOG_FILE=${ENVTRTDIR}/build_head1_gdb.log
elif [ "$ENGINE_TYPE" == "head2" ]; then
    ONNX_FILE=${ENV_HEAD2_ONNX}
    ENGINE_FILE=${ENV_HEAD2_ENGINE}
    LOG_FILE=${ENVTRTDIR}/build_head2_gdb.log
else
    echo "错误: 不支持的引擎类型 '$ENGINE_TYPE'"
    echo "支持的引擎类型: head1, head2"
    exit 1
fi

echo "ONNX文件: $ONNX_FILE"
echo "引擎文件: $ENGINE_FILE"
echo "日志文件: $LOG_FILE"
echo ""

# 创建GDB命令脚本
GDB_SCRIPT=$(mktemp /tmp/gdb_script.XXXXXX)
cat > "$GDB_SCRIPT" << 'EOF'
# GDB调试脚本
set pagination off
set logging file /tmp/gdb_output.log
set logging on

# 设置断点（可选）
# break DeformableAttentionAggrPlugin::supportsFormatCombination
# break DeformableAttentionAggrPlugin::getOutputDimensions
# break DeformableAttentionAggrPlugin::getWorkspaceSize
# break DeformableAttentionAggrPlugin::getOutputDataType

# 运行程序
run

# 如果发生段错误，打印堆栈跟踪
if $_siginfo
    echo \n
    echo ================ 段错误堆栈跟踪 ================\n
    backtrace
    echo \n
    echo ================ 详细堆栈信息 ================\n
    backtrace full
    echo \n
    echo ================ 寄存器信息 ================\n
    info registers
    echo \n
    echo ================ 内存映射 ================\n
    info proc mappings
    echo \n
    echo ================ 线程信息 ================\n
    info threads
    echo \n
    echo ================ 当前线程堆栈 ================\n
    thread apply all backtrace
    echo \n
    quit
end

continue
EOF

echo "启动GDB调试..."
echo "GDB脚本: $GDB_SCRIPT"
echo ""

# 使用GDB运行trtexec
gdb -batch -x "$GDB_SCRIPT" \
    --args ${ENV_TensorRT_BIN}/trtexec \
    --onnx=${ONNX_FILE} \
    ${PLUGIN_ARGS} \
    --memPoolSize=workspace:2048 \
    --saveEngine=${ENGINE_FILE} \
    --verbose \
    --warmUp=1 \
    --iterations=1 \
    --dumpOutput \
    --dumpProfile \
    --dumpLayerInfo \
    --exportOutput=${ENVTRTDIR}/buildOutput_${ENGINE_TYPE}.json \
    --exportProfile=${ENVTRTDIR}/buildProfile_${ENGINE_TYPE}.json \
    --exportLayerInfo=${ENVTRTDIR}/buildLayerInfo_${ENGINE_TYPE}.json \
    --profilingVerbosity=detailed \
    ${PRECISION_ARGS} \
    > ${LOG_FILE} 2>&1

GDB_EXIT_CODE=$?

echo ""
echo "===================================================================================================================="
echo "GDB调试完成，退出代码: $GDB_EXIT_CODE"
echo "详细日志: $LOG_FILE"
echo "GDB输出: /tmp/gdb_output.log"
echo "===================================================================================================================="

# 显示GDB输出
if [ -f /tmp/gdb_output.log ]; then
    echo ""
    echo "======== GDB输出 ========"
    cat /tmp/gdb_output.log
    echo ""
fi

# 清理临时文件
rm -f "$GDB_SCRIPT"

exit $GDB_EXIT_CODE

