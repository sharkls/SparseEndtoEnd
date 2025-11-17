#!/usr/bin/env bash

set -euo pipefail

# 用法：
#   bash deploy/tools/profile_head2.sh [--fp16|--fp32|--int8]
# 说明：
#   - 自动加载环境变量，使用已构建的 Head2 引擎进行 trtexec 逐层 profile；
#   - 结果保存到 deploy/profiles/ 下（含 json 与 log）；

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEPLOY_DIR="$(dirname "$SCRIPT_DIR")"

# 1) 环境准备
cd "$DEPLOY_DIR"
mkdir -p profiles

# 注意：set -u 会让被 source 的脚本在引用未定义变量时报错。
# tools/set_env.sh 里可能在赋值前 echo/引用了变量名，
# 因此在 source 前临时关闭 -u，source 完再恢复。
set +u
source tools/set_env.sh
set -u

PREC_FLAG="--fp16"
IO_FMT_ARGS="--inputIOFormats=fp16:chw --outputIOFormats=fp16:chw"
if [[ "${1:-}" == "--fp32" ]]; then
  PREC_FLAG=""
  IO_FMT_ARGS=""
elif [[ "${1:-}" == "--int8" ]]; then
  PREC_FLAG="--int8 --strictTypeConstraints"
  # 如需为 INT8 指定 I/O 格式，请根据实际 engine 的格式支持补充；此处保持默认以避免不匹配
  IO_FMT_ARGS=""
fi

STAMP="$(date +%Y%m%d_%H%M%S)"
PROFILE_JSON="profiles/head2_profile_${STAMP}.json"
PROFILE_LOG="profiles/trtexec_head2_${STAMP}.log"

# 2) 基本检查
if [[ ! -f "$ENV_HEAD2_ENGINE" ]]; then
  echo "[ERROR] Head2 engine not found: $ENV_HEAD2_ENGINE" >&2
  echo "请先执行: bash deploy/build_sparse4d_engine.sh fp16" >&2
  exit 1
fi

if [[ ! -f "$ENVTARGETPLUGIN" ]]; then
  echo "[ERROR] Plugin not found: $ENVTARGETPLUGIN" >&2
  exit 1
fi

echo "[INFO] Using engine: $ENV_HEAD2_ENGINE"
echo "[INFO] Using plugin: $ENVTARGETPLUGIN"
echo "[INFO] Saving profile to: $PROFILE_JSON"
echo "[INFO] Saving log to: $PROFILE_LOG"

# 3) 执行 trtexec（逐层 profile）
"${ENV_TensorRT_BIN}/trtexec" \
  --loadEngine="${ENV_HEAD2_ENGINE}" \
  --plugins="${ENVTARGETPLUGIN}" \
  --dumpProfile \
  --exportProfile="${PROFILE_JSON}" \
  --separateProfileRun \
  --warmUp=1000 \
  --iterations=200 \
  --useCudaGraph \
  --useSpinWait \
  ${IO_FMT_ARGS} \
  --verbose \
  ${PREC_FLAG} \
  2>&1 | tee "${PROFILE_LOG}"

echo "\n[INFO] Top 重排相关节点（ForeignNode/Reformat/Shuffle/Transpose）:" | tee -a "${PROFILE_LOG}"
grep -E "ForeignNode|Reformat|Shuffle|Transpose" "${PROFILE_LOG}" | head -200 | tee -a "${PROFILE_LOG}" || true

echo "\n[INFO] Done. Profile JSON: ${PROFILE_JSON}" | tee -a "${PROFILE_LOG}"


