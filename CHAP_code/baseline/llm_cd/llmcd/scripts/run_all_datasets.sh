#!/bin/bash
# ============================================================
# 跑完所有数据集的 LLM-CD baseline（三步：PC → Graph → Train）
# 用法：
#   chmod +x baseline/llmcd/scripts/run_all_datasets.sh
#   bash baseline/llmcd/scripts/run_all_datasets.sh
#
# GPU 集群可以用 Slurm 版本（见下方 run_all_slurm.sh）
# ============================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

# ---- 配置 ----
PYTHON=".venv/bin/python"
RUNS_DIR="baseline/llmcd/runs"
CONFIG_DIR="baseline/llmcd/configs"

# 所有要跑的数据集
DATASETS=(
  "synthetic"
  "adult"
  "cardio"
  "creditcard"
  "crime"
  "diamonds"
  "diamonds_mixed"
  "elevator"
  "housesale"
  "housing"
  "meps"
)

mkdir -p "$RUNS_DIR"

for ds in "${DATASETS[@]}"; do
  CONFIG="$CONFIG_DIR/${ds}.json"
  RUN_DIR="$RUNS_DIR/${ds}_llmcd"

  # 检查配置文件是否存在
  if [ ! -f "$CONFIG" ]; then
    echo "[SKIP] $ds — 配置文件 $CONFIG 不存在"
    continue
  fi

  echo "============================================"
  echo "  $ds"
  echo "============================================"

  # ---- Stage 1: PC 因果发现 ----
  echo "  [1/3] PC discovery..."
  $PYTHON baseline/llmcd/discover_pc_graph.py \
    --config "$CONFIG" \
    --run_dir "$RUN_DIR"

  # ---- Stage 2: 图判定（无 LLM，fallback 规则） ----
  echo "  [2/3] Graph judgment (fallback)..."
  $PYTHON baseline/llmcd/judge_llm_graph.py \
    --config "$CONFIG" \
    --pc_graph "$RUN_DIR/pc_graph.json" \
    --run_dir "$RUN_DIR"

  # ---- Stage 3: 父节点训练 & 评估 ----
  echo "  [3/3] Train & eval..."
  $PYTHON baseline/llmcd/train_eval_from_graph.py \
    --config "$CONFIG" \
    --graph "$RUN_DIR/final_graph.json" \
    --run_dir "$RUN_DIR"

  echo "  Done → $RUN_DIR/metrics.json"
  echo ""
done

echo "============================================"
echo "  全部跑完！结果汇总："
echo "============================================"
for ds in "${DATASETS[@]}"; do
  RUN_DIR="$RUNS_DIR/${ds}_llmcd"
  if [ -f "$RUN_DIR/metrics.json" ]; then
    echo "  $ds ✅"
  else
    echo "  $ds ❌ (无 metrics)"
  fi
done
