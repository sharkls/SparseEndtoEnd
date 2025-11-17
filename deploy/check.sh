#!/usr/bin/env bash
# run_head2_profile.sh - 用于分析 Head2 引擎的逐层耗时
set -euo pipefail

# 0) GPU 性能模式优化（减少波动）
# 设置 GPU 为性能模式，固定频率（如果支持）
if command -v nvidia-smi &> /dev/null; then
    # 尝试设置 GPU 为性能模式
    GPU_ID="${CUDA_VISIBLE_DEVICES:-0}"
    nvidia-smi -i ${GPU_ID} -pm 1 2>/dev/null || true  # 启用持久化模式
    nvidia-smi -i ${GPU_ID} -pl 100 2>/dev/null || true  # 设置最大功耗限制（如果支持）
    # 注意：固定频率需要 root 权限，这里仅做尝试
fi

# 设置环境变量减少波动
export CUDA_LAUNCH_BLOCKING=0  # 保持异步执行以获得真实性能
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_CACHE_DISABLE=0  # 保持缓存以提高稳定性
# 设置 TensorRT 线程数（避免 CPU 调度影响）
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# 1) 路径配置（按需修改）
ENGINE="${ENGINE:-/share/Code/Sparse4dE2E/deploy/engine/sparse4dhead2nd.engine}"
PLUGIN="${PLUGIN:-/share/Code/Sparse4dE2E/deploy/dfa_plugin/lib/deformableAttentionAggr.so}"
OUT_DIR="${OUT_DIR:-/share/Code/Sparse4dE2E/deploy/profiles}"
mkdir -p "${OUT_DIR}"

# 2) 精度标记（仅用于命名文件，不影响 engine）
PREC="${1:-fp32}"   # 可传 fp32 / fp16 / int8 等，用于区分输出文件名
TS="$(date +%Y%m%d_%H%M%S)"

# 3) 形状（与 ONNX 模型的实际输入形状保持一致）
# 注意：名称和维度需与导出的 ONNX 模型输入名严格一致
# 基于实际 ONNX 模型输入形状：
#   feature: [1, 89760, 256]
#   spatial_shapes: [6, 4, 2]
#   level_start_index: [6, 4]
#   instance_feature: [1, 900, 256]
#   anchor: [1, 900, 11]
#   time_interval: [1]
#   temp_instance_feature: [1, 600, 256]
#   temp_anchor: [1, 600, 11]
#   mask: [1]
#   track_id: [1, 900]
#   image_wh: [1, 6, 2]
#   lidar2img: [1, 6, 4, 4]
SHAPES="feature:1x89760x256,\
spatial_shapes:6x4x2,level_start_index:6x4,instance_feature:1x900x256,anchor:1x900x11,\
time_interval:1,temp_instance_feature:1x600x256,temp_anchor:1x600x11,mask:1,track_id:1x900,\
image_wh:1x6x2,lidar2img:1x6x4x4"

# 4) 输出文件
LOG_FILE="${OUT_DIR}/trtexec_${PREC}_${TS}.log"
PROFILE_JSON="${OUT_DIR}/profile_${PREC}_${TS}.json"

# 5) 运行 trtexec（启用逐层耗时分析）
# 参数说明：
#   --loadEngine: 加载预构建的 TensorRT 引擎
#   --plugins: 加载自定义插件（DeformableAttentionAggr）
#   --shapes: 指定输入形状（必须与 ONNX 模型输入形状匹配）
#   --dumpProfile: 输出逐层耗时到控制台
#   --separateProfileRun: 单独运行一次用于性能分析
#   --noDataTransfers: 禁用显式数据搬运统计，专注算子计算时间
#   --exportProfile: 导出逐层耗时到 JSON 文件
#   --iterations: 性能测试迭代次数（增加以获得更稳定的平均值）
#   --warmUp: 预热迭代次数（让 GPU 达到稳定状态，包括频率稳定）
#   --streams: CUDA 流数量
#   --avgRuns: 多次运行取平均值（减少波动）
#   --useSpinWait: 使用自旋等待而非睡眠，减少 CPU 调度延迟
# 注意：如遇到 crash，可先去掉 --noDataTransfers 再试
CMD=(trtexec
  --loadEngine="${ENGINE}"
  --plugins="${PLUGIN}"
  --shapes="${SHAPES}"
  --dumpProfile
  --separateProfileRun
  --noDataTransfers
  --exportProfile="${PROFILE_JSON}"
  --iterations=1000
  --warmUp=1000
  --useCudaGraph
  --useSpinWait
  --streams=1
)

echo "[INFO] Running: ${CMD[*]}"
"${CMD[@]}" 2>&1 | tee "${LOG_FILE}"

echo "[INFO] Done."
echo "[INFO] Log: ${LOG_FILE}"
echo "[INFO] Per-layer JSON: ${PROFILE_JSON}"