#!/bin/bash
# 使用core dump调试段错误

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
echo "启用core dump调试"
echo "精度: $PRECISION"
echo "===================================================================================================================="

# 选择要调试的引擎
ENGINE_TYPE=${2:-head1}

if [ "$ENGINE_TYPE" == "head1" ]; then
    ONNX_FILE=${ENV_HEAD1_ONNX}
    ENGINE_FILE=${ENV_HEAD1_ENGINE}
    LOG_FILE=${ENVTRTDIR}/build_head1.log
elif [ "$ENGINE_TYPE" == "head2" ]; then
    ONNX_FILE=${ENV_HEAD2_ONNX}
    ENGINE_FILE=${ENV_HEAD2_ENGINE}
    LOG_FILE=${ENVTRTDIR}/build_head2.log
else
    echo "错误: 不支持的引擎类型 '$ENGINE_TYPE'"
    exit 1
fi

# 启用core dump
ulimit -c unlimited
echo "Core dump已启用 (ulimit -c: $(ulimit -c))"

# 设置core dump文件名模式
echo "core.%e.%p.%t" > /proc/sys/kernel/core_pattern 2>/dev/null || echo "无法设置core_pattern（需要root权限）"

# 运行trtexec（会生成core dump）
echo "运行trtexec..."
${ENV_TensorRT_BIN}/trtexec \
    --onnx=${ONNX_FILE} \
    ${PLUGIN_ARGS} \
    --memPoolSize=workspace:2048 \
    --saveEngine=${ENGINE_FILE} \
    --verbose \
    --warmUp=1 \
    --iterations=1 \
    ${PRECISION_ARGS} \
    > ${LOG_FILE} 2>&1

EXIT_CODE=$?

# 查找core dump文件
CORE_FILE=$(find . -name "core.*" -type f -newer ${LOG_FILE} 2>/dev/null | head -1)

if [ -n "$CORE_FILE" ]; then
    echo ""
    echo "===================================================================================================================="
    echo "发现core dump文件: $CORE_FILE"
    echo "使用GDB分析core dump..."
    echo "===================================================================================================================="
    echo ""
    
    # 使用GDB分析core dump
    gdb -batch -ex "thread apply all backtrace full" \
        -ex "info registers" \
        -ex "info proc mappings" \
        ${ENV_TensorRT_BIN}/trtexec "$CORE_FILE" 2>&1 | tee ${ENVTRTDIR}/gdb_coredump_${ENGINE_TYPE}.log
    
    echo ""
    echo "GDB分析结果已保存到: ${ENVTRTDIR}/gdb_coredump_${ENGINE_TYPE}.log"
else
    echo "未找到core dump文件"
    echo "如果程序崩溃但没有生成core dump，请检查："
    echo "1. ulimit -c 是否设置为unlimited"
    echo "2. /proc/sys/kernel/core_pattern 是否允许写入"
    echo "3. 当前目录是否有写权限"
fi

exit $EXIT_CODE

