#!/bin/bash
# GDB调试脚本：专门用于分析段错误
# 使用方法: ./debug_segfault.sh [fp16|fp32] [head1|head2]

cd /share/Code/Sparse4dE2E/deploy
source tools/set_env.sh

# 解析参数
PRECISION=${1:-fp16}
ENGINE_TYPE=${2:-head1}

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
    PLUGIN_ARGS="${PLUGIN_ARGS} --plugins=$(pwd)/dfa_plugin/lib/deformableAttentionAggr.so"
    echo "[INFO] DeformableAttentionAggrPlugin enabled: $(pwd)/dfa_plugin/lib/deformableAttentionAggr.so"
fi
if [[ -n "${ENV_LAYER_NORM_PLUGIN}" && -f "${ENV_LAYER_NORM_PLUGIN}" ]]; then
    PLUGIN_ARGS="${PLUGIN_ARGS} --plugins=$(pwd)/ln_plugin/lib/customLayerNorm.so"
    echo "[INFO] LayerNormPlugin enabled: $(pwd)/ln_plugin/lib/customLayerNorm.so"
fi
if [[ -n "${ENV_SPARSEBOX_PLUGIN}" && -f "${ENV_SPARSEBOX_PLUGIN}" ]]; then
    PLUGIN_ARGS="${PLUGIN_ARGS} --plugins=$(pwd)/sparsebox_plugin/lib/SparseBox3DKeyPointsPlugin.so"
    echo "[INFO] SparseBox3DKeyPointsPlugin enabled: $(pwd)/sparsebox_plugin/lib/SparseBox3DKeyPointsPlugin.so"
fi

# 选择要调试的引擎
if [ "$ENGINE_TYPE" == "head1" ]; then
    ONNX_FILE=${ENV_HEAD1_ONNX}
    ENGINE_FILE=${ENV_HEAD1_ENGINE}
    LOG_FILE=${ENVTRTDIR}/debug_segfault_head1.log
elif [ "$ENGINE_TYPE" == "head2" ]; then
    ONNX_FILE=${ENV_HEAD2_ONNX}
    ENGINE_FILE=${ENV_HEAD2_ENGINE}
    LOG_FILE=${ENVTRTDIR}/debug_segfault_head2.log
else
    echo "错误: 不支持的引擎类型 '$ENGINE_TYPE'"
    exit 1
fi

echo "===================================================================================================================="
echo "GDB段错误调试分析"
echo "精度: $PRECISION"
echo "引擎: $ENGINE_TYPE"
echo "ONNX: $ONNX_FILE"
echo "===================================================================================================================="

# 创建GDB命令脚本
GDB_SCRIPT=$(mktemp /tmp/gdb_segfault.XXXXXX)
cat > "$GDB_SCRIPT" << 'GDBEOF'
# GDB段错误分析脚本
set pagination off
set print pretty on
set print elements 0
set print null-stop on
set breakpoint pending on

# 设置断点监控关键函数
break DeformableAttentionAggrPlugin::supportsFormatCombination
commands
    printf "\n[断点] supportsFormatCombination: pos=%d, nbInputs=%d, nbOutputs=%d\n", pos, nbInputs, nbOutputs
    if pos == 3 || pos == 4
        printf "  [混合精度检查] valueType=%d, keypointType=%d\n", inOut[0].type, inOut[pos].type
    end
    continue
end

break DeformableAttentionAggrPlugin::getWorkspaceSize
commands
    printf "\n[断点] getWorkspaceSize: nbInputs=%d, nbOutputs=%d\n", nbInputs, nbOutputs
    if nbInputs >= 1
        printf "  inputs[0].type = %d\n", inputs[0].type
    end
    continue
end

break DeformableAttentionAggrPlugin::getOutputDimensions
commands
    printf "\n[断点] getOutputDimensions: outputIndex=%d, nbInputs=%d\n", outputIndex, nbInputs
    continue
end

break DeformableAttentionAggrPlugin::getOutputDataType
commands
    printf "\n[断点] getOutputDataType: index=%d, nbInputs=%d\n", index, nbInputs
    continue
end

# 运行程序
run

# 如果发生段错误
if $_siginfo
    echo \n
    echo ========================================
    echo 段错误分析报告
    echo ========================================\n
    
    echo [1] 堆栈跟踪（完整）\n
    backtrace full
    
    echo \n[2] 当前帧信息\n
    info frame
    
    echo \n[3] 当前帧的局部变量\n
    info locals
    
    echo \n[4] 当前帧的函数参数\n
    info args
    
    echo \n[5] 寄存器信息\n
    info registers
    
    echo \n[6] 所有线程的堆栈\n
    thread apply all backtrace
    
    echo \n[7] 内存映射（前20行）\n
    info proc mappings | head -20
    
    echo \n[8] 检查关键指针\n
    if $pc != 0
        printf "程序计数器 (PC): 0x%lx\n", $pc
        x/10i $pc-20
    end
    
    echo \n[9] 检查插件相关符号\n
    info symbol $pc
    
    echo \n[10] 查看最近调用的函数\n
    frame 0
    if $argc > 0
        printf "函数参数数量: %d\n", $argc
    end
    
    echo \n========================================
    echo 分析完成
    echo ========================================\n
    
    quit
end

continue
GDBEOF

# 运行GDB
echo "启动GDB调试..."
gdb -batch -x "$GDB_SCRIPT" \
    --args ${ENV_TensorRT_BIN}/trtexec \
    --onnx=${ONNX_FILE} \
    ${PLUGIN_ARGS} \
    --memPoolSize=workspace:2048 \
    --saveEngine=${ENGINE_FILE} \
    --verbose \
    --warmUp=1 \
    --iterations=1 \
    ${PRECISION_ARGS} \
    2>&1 | tee ${LOG_FILE}

GDB_EXIT_CODE=$?

echo ""
echo "===================================================================================================================="
echo "GDB调试完成，退出代码: $GDB_EXIT_CODE"
echo "详细日志已保存到: ${LOG_FILE}"
echo "===================================================================================================================="

# 清理
rm -f "$GDB_SCRIPT"

# 分析日志中的关键信息
if [ -f "${LOG_FILE}" ]; then
    echo ""
    echo "========== 关键信息提取 =========="
    echo ""
    
    echo "[段错误位置]"
    grep -A 5 "段错误\|Segmentation\|SIGSEGV" ${LOG_FILE} | head -10
    
    echo ""
    echo "[堆栈跟踪]"
    grep -A 20 "backtrace\|#0\|#1\|#2\|#3\|#4\|#5" ${LOG_FILE} | head -30
    
    echo ""
    echo "[插件函数调用]"
    grep -E "supportsFormatCombination|getWorkspaceSize|getOutputDimensions|DeformableAttentionAggrPlugin" ${LOG_FILE} | tail -20
    
    echo ""
    echo "[混合精度相关]"
    grep -E "混合精度|valueType|keypointType|FP16|FP32|HALF|FLOAT" ${LOG_FILE} | tail -10
fi

exit $GDB_EXIT_CODE

