#!/usr/bin/env bash
# run_head2_profile.sh
set -euo pipefail

# 1) 路径配置（按需修改）
ENGINE="${ENGINE:-/share/Code/Sparse4dE2E/deploy/engine/engine8/sparse4dhead2nd.engine}"
PLUGIN="${PLUGIN:-/share/Code/Sparse4dE2E/deploy/dfa_plugin/lib/deformableAttentionAggr.so}"
OUT_DIR="${OUT_DIR:-/share/Code/Sparse4dE2E/deploy/profiles}"
mkdir -p "${OUT_DIR}"

# 2) 精度标记（仅用于命名文件，不影响 engine）
PREC="${1:-fp32}"   # 可传 fp32 / fp16 / int8 等，用于区分输出文件名
TS="$(date +%Y%m%d_%H%M%S)"

# 3) 形状（与 engine 的输入保持一致；若是扁平 1D，请改用下面的 FLAT 形状）
# 注意：名称需与导出的 engine 输入名严格一致
SHAPES="feature:1x89760x256,\
spatial_shapes:1x48,level_start_index:1x24,instance_feature:1x900x256,anchor:1x900x11,\
time_interval:1x1,temp_instance_feature:1x600x256,temp_anchor:1x600x11,mask:1x1,track_id:1x900,\
image_wh:1x12,lidar2img:1x96"

# 如果 engine 是扁平一维输入（请改成你实际的输入名），取消注释以下一行并注释上面的 SHAPES：
# SHAPES="feature:1x22940160,spatial_shapes:1x48,level_start_index:1x24,instance_feature:1x230400,anchor:1x9900,time_interval:1x1,temp_instance_feature:1x153600,temp_anchor:1x6600,mask:1x1,track_id:1x900,image_wh:1x12,lidar2img:1x96"

# 4) 输出文件
LOG_FILE="${OUT_DIR}/trtexec_${PREC}_${TS}.log"
PROFILE_JSON="${OUT_DIR}/profile_${PREC}_${TS}.json"

# 5) 运行 trtexec（启用逐层耗时；单流；禁用显式数据搬运以专注算子时间线上）
# 如遇到 crash，可先去掉 --noDataTransfers 再试
CMD=(trtexec
  --loadEngine="${ENGINE}"
  --plugins="${PLUGIN}"
  --shapes="${SHAPES}"
  --dumpProfile
  --separateProfileRun
  --noDataTransfers
  --exportProfile="${PROFILE_JSON}"
  --iterations=10
  --warmUp=200
  --streams=1
)

echo "[INFO] Running: ${CMD[*]}"
"${CMD[@]}" 2>&1 | tee "${LOG_FILE}"

echo "[INFO] Done."
echo "[INFO] Log: ${LOG_FILE}"
echo "[INFO] Per-layer JSON: ${PROFILE_JSON}"