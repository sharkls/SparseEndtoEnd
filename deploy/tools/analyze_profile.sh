#!/usr/bin/env bash
# Profile分析工具
# 用法: bash deploy/tools/analyze_profile.sh <profile_log_file>

set -euo pipefail

if [ $# -lt 1 ]; then
    echo "用法: $0 <profile_log_file>"
    echo "示例: $0 deploy/profiles/trtexec_head2_20251107_063943.log"
    exit 1
fi

PROFILE_LOG="$1"

if [ ! -f "$PROFILE_LOG" ]; then
    echo "错误: 文件不存在: $PROFILE_LOG"
    exit 1
fi

echo "=========================================="
echo "Profile分析报告: $(basename $PROFILE_LOG)"
echo "=========================================="
echo ""

# 1. 提取总体性能指标
echo "【1. 总体性能指标】"
echo "----------------------------------------"
if grep -q "Performance summary" "$PROFILE_LOG"; then
    grep -A 8 "Performance summary" "$PROFILE_LOG" | grep -E "GPU Compute Time|H2D Latency|Total|Throughput|Latency.*mean" | head -6
else
    echo "未找到性能摘要"
fi
echo ""

# 2. 提取最耗时的节点（Top 20）
echo "【2. 最耗时的节点 (Top 20)】"
echo "----------------------------------------"
echo "节点名称 | 总时间(ms) | 平均时间(ms) | 占比(%)"
echo "----------------------------------------"
grep "Time %" "$PROFILE_LOG" | \
    awk '{
        # 提取Time %（最后一列）
        time_pct = $NF
        if (time_pct+0 > 0) {
            # 提取平均时间（倒数第二列）
            avg_time = $(NF-1)
            # 提取总时间（倒数第三列）
            total_time = $(NF-2)
            # 提取节点名称（前面的所有列，去掉最后三列）
            node_name = ""
            for (i=1; i<=NF-3; i++) {
                if (i>1) node_name = node_name " "
                node_name = node_name $i
            }
            printf "%.1f%% | %.2f ms | %s\n", time_pct, avg_time, node_name
        }
    }' | \
    sort -rn | head -20
echo ""

# 3. ForeignNode节点分析
echo "【3. ForeignNode节点分析（最大瓶颈）】"
echo "----------------------------------------"
FOREIGNNODE_COUNT=$(grep -c "ForeignNode" "$PROFILE_LOG" || echo "0")
if [ "$FOREIGNNODE_COUNT" -gt 0 ]; then
    echo "ForeignNode节点数量: $FOREIGNNODE_COUNT"
    echo ""
    echo "节点详情:"
    grep "ForeignNode" "$PROFILE_LOG" | \
        awk '{
            time_pct = $NF
            avg_time = $(NF-1)
            total_time = $(NF-2)
            node_name = ""
            for (i=1; i<=NF-3; i++) {
                if (i>1) node_name = node_name " "
                node_name = node_name $i
            }
            printf "  %.1f%% (%.2f ms) - %s\n", time_pct, avg_time, node_name
        }' | \
        sort -rn
    echo ""
    # 计算总占比
    FOREIGNNODE_TOTAL=$(grep "ForeignNode" "$PROFILE_LOG" | \
        awk '{sum+=$NF} END {printf "%.1f", sum}')
    echo "ForeignNode总占比: ${FOREIGNNODE_TOTAL}%"
    if (( $(echo "$FOREIGNNODE_TOTAL > 30" | bc -l) )); then
        echo "⚠️  警告: ForeignNode占比过高（>30%），需要优化！"
    fi
else
    echo "未找到ForeignNode节点"
fi
echo ""

# 4. Reformat节点分析
echo "【4. Reformat节点分析】"
echo "----------------------------------------"
REFORMAT_COUNT=$(grep -c "Reformat" "$PROFILE_LOG" || echo "0")
if [ "$REFORMAT_COUNT" -gt 0 ]; then
    echo "Reformat节点数量: $REFORMAT_COUNT"
    echo ""
    echo "主要Reformat节点:"
    grep "Reformat" "$PROFILE_LOG" | \
        awk '{
            time_pct = $NF
            if (time_pct+0 > 0.1) {  # 只显示占比>0.1%的
                avg_time = $(NF-1)
                node_name = ""
                for (i=1; i<=NF-3; i++) {
                    if (i>1) node_name = node_name " "
                    node_name = node_name $i
                }
                printf "  %.2f%% (%.3f ms) - %s\n", time_pct, avg_time, node_name
            }
        }' | \
        sort -rn | head -10
    echo ""
    # 计算总占比
    REFORMAT_TOTAL=$(grep "Reformat" "$PROFILE_LOG" | \
        awk '{sum+=$NF} END {printf "%.1f", sum}')
    echo "Reformat总占比: ${REFORMAT_TOTAL}%"
    if (( $(echo "$REFORMAT_TOTAL > 2" | bc -l) )); then
        echo "⚠️  警告: Reformat占比较高（>2%），可以优化"
    fi
else
    echo "未找到Reformat节点"
fi
echo ""

# 5. Plugin节点分析
echo "【5. DeformableAttentionAggrPlugin节点分析】"
echo "----------------------------------------"
PLUGIN_COUNT=$(grep -c "DeformableAttentionAggrPlugin[^_]" "$PROFILE_LOG" || echo "0")
if [ "$PLUGIN_COUNT" -gt 0 ]; then
    echo "Plugin实例数量: $PLUGIN_COUNT"
    echo ""
    echo "Plugin详情:"
    grep "DeformableAttentionAggrPlugin[^_]" "$PROFILE_LOG" | \
        awk '{
            time_pct = $NF
            avg_time = $(NF-1)
            total_time = $(NF-2)
            node_name = ""
            for (i=1; i<=NF-3; i++) {
                if (i>1) node_name = node_name " "
                node_name = node_name $i
            }
            printf "  %.1f%% (%.2f ms) - %s\n", time_pct, avg_time, node_name
        }' | \
        sort -rn
    echo ""
    # 计算总占比
    PLUGIN_TOTAL=$(grep "DeformableAttentionAggrPlugin[^_]" "$PROFILE_LOG" | \
        awk '{sum+=$NF} END {printf "%.1f", sum}')
    echo "Plugin总占比: ${PLUGIN_TOTAL}%"
else
    echo "未找到Plugin节点"
fi
echo ""

# 6. 优化建议
echo "【6. 优化建议】"
echo "----------------------------------------"
FOREIGNNODE_TOTAL=${FOREIGNNODE_TOTAL:-0}
REFORMAT_TOTAL=${REFORMAT_TOTAL:-0}

if (( $(echo "$FOREIGNNODE_TOTAL > 30" | bc -l 2>/dev/null || echo "0") )); then
    echo "🔴 高优先级: ForeignNode占比 ${FOREIGNNODE_TOTAL}%"
    echo "   建议: 优化permute/transpose操作，减少ForeignNode节点"
    echo "   方法: 在export_head_onnx.py中优化points_2d和weights的permute"
    echo ""
fi

if (( $(echo "$REFORMAT_TOTAL > 2" | bc -l 2>/dev/null || echo "0") )); then
    echo "🟡 中优先级: Reformat占比 ${REFORMAT_TOTAL}%"
    echo "   建议: 统一数据类型和内存布局，减少Reformat节点"
    echo "   方法: 启用layout优化，使用IO格式参数"
    echo ""
fi

if (( $(echo "$FOREIGNNODE_TOTAL <= 30 && $REFORMAT_TOTAL <= 2" | bc -l 2>/dev/null || echo "0") )); then
    echo "✅ 当前性能良好，主要瓶颈已优化"
fi

echo ""
echo "=========================================="
echo "分析完成"
echo "=========================================="

