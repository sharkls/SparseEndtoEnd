#!/usr/bin/env bash
set -euo pipefail

# 工程根路径（绝对路径）
ROOT="/share/Code/SparseEnd2End"
SRC="$ROOT/C++/script/model_input/topkinstance.cpp"
OUTDIR="$ROOT/C++/Output/testAlgLib-topk"
BIN="$OUTDIR/topkinstance"

# 默认参数
DEFAULT_BIN_PATH="$ROOT/C++/Output/val_bin/sample_0_pred_class_score_1*900*10_float32.bin"
DEFAULT_TOPK=600

# 读取用户参数
BIN_PATH="${1:-$DEFAULT_BIN_PATH}"
TOPK="${2:-$DEFAULT_TOPK}"

mkdir -p "$OUTDIR"

echo "[BUILD] g++ $SRC -> $BIN"
g++ -std=c++17 -O2 "$SRC" -o "$BIN"

echo "[RUN ] $BIN \"$BIN_PATH\" $TOPK"
"$BIN" "$BIN_PATH" "$TOPK" 