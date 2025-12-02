#!/bin/bash
# 文件名: debug_with_breakpoints.sh

cd /share/Code/Sparse4dE2E/deploy
source tools/set_env.sh

# 创建GDB命令脚本
GDB_SCRIPT=$(mktemp /tmp/gdb_breakpoints.XXXXXX)
cat > "$GDB_SCRIPT" << 'GDBEOF'
# GDB调试脚本 - 混合精度模式调试
set pagination off
set print pretty on
set print elements 0

# 允许pending breakpoint（在库加载后自动设置）
set breakpoint pending on

# ========== 设置断点（使用pending模式）==========
# 1. supportsFormatCombination - 检查格式组合
break DeformableAttentionAggrPlugin::supportsFormatCombination
commands
    echo \n========== supportsFormatCombination 被调用 ==========\n
    print pos
    print nbInputs
    print nbOutputs
    if pos == 3 || pos == 4
        echo \n*** 检查位置3或4（keypoints）***\n
        if nbInputs >= 1
            print inOut[0].type
            print inOut[pos].type
        end
    end
    continue
end

# 2. 使用文件行号设置断点（更可靠）
break deformableAttentionAggrPlugin.cpp:160
commands
    echo \n========== 到达混合精度检查代码 ==========\n
    print pos
    print nbInputs
    if nbInputs >= 1
        print inOut[0].type
        print inOut[pos].type
    end
    continue
end

# 3. getWorkspaceSize
break deformableAttentionAggrPlugin.cpp:220
commands
    echo \n========== getWorkspaceSize 被调用 ==========\n
    print nbInputs
    print nbOutputs
    if nbInputs >= 1
        print inputs[0].type
        if nbInputs >= 5
            print inputs[3].type
            print inputs[4].type
        end
    end
    continue
end

# 运行程序
run

# 如果发生段错误
if $_siginfo
    echo \n
    echo ================ 段错误发生 ================\n
    backtrace full
    echo \n
    echo ================ 当前帧信息 ================\n
    info frame
    echo \n
    echo ================ 局部变量 ================\n
    info locals
    echo \n
    echo ================ 函数参数 ================\n
    info args
    quit
end

continue
GDBEOF

# 使用GDB运行
gdb -batch -x "$GDB_SCRIPT" \
    --args ${ENV_TensorRT_BIN}/trtexec \
    --onnx=${ENV_HEAD1_ONNX} \
    --plugins=$(pwd)/dfa_plugin/lib/deformableAttentionAggr.so \
    --plugins=$(pwd)/ln_plugin/lib/customLayerNorm.so \
    --plugins=$(pwd)/sparsebox_plugin/lib/SparseBox3DKeyPointsPlugin.so \
    --memPoolSize=workspace:2048 \
    --saveEngine=${ENV_HEAD1_ENGINE} \
    --fp16 \
    --verbose \
    --warmUp=1 \
    --iterations=1 \
    2>&1 | tee /tmp/gdb_breakpoints_output.log

# 清理
rm -f "$GDB_SCRIPT"