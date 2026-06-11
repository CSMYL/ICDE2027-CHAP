#!/bin/bash
# ============================================================
# Slurm 数组任务 — 每个数据集一个 job
# 用法：sbatch baseline/llmcd/scripts/run_all_slurm.sh
# ============================================================
#SBATCH --job-name=llmcd_baseline
#SBATCH --output=logs/llmcd_%A_%a.out
#SBATCH --error=logs/llmcd_%A_%a.err
#SBATCH --array=0-10
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --partition=gpu    # 按你的集群改

set -e

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

# ---- Slurm 会为每个 array index 跑一个 job ----
TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
DS="${DATASETS[$TASK_ID]}"

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$PROJECT_ROOT"

mkdir -p logs
mkdir -p "baseline/llmcd/runs"

CONFIG="baseline/llmcd/configs/${DS}.json"
RUN_DIR="baseline/llmcd/runs/${DS}_llmcd"

if [ ! -f "$CONFIG" ]; then
    echo "[SKIP] $DS — config not found"
    exit 0
fi

echo "============================================"
echo "  Dataset: $DS  (task $TASK_ID)"
echo "============================================"

source .venv/bin/activate

# ---- Stage 1: PC ----
echo "[1/3] PC discovery..."
python baseline/llmcd/discover_pc_graph.py \
    --config "$CONFIG" \
    --run_dir "$RUN_DIR"

# ---- Stage 2: Fallback graph ----
echo "[2/3] Graph judgment..."
python baseline/llmcd/judge_llm_graph.py \
    --config "$CONFIG" \
    --pc_graph "$RUN_DIR/pc_graph.json" \
    --run_dir "$RUN_DIR"

# ---- Stage 3: Train & eval ----
echo "[3/3] Train & eval..."
python baseline/llmcd/train_eval_from_graph.py \
    --config "$CONFIG" \
    --graph "$RUN_DIR/final_graph.json" \
    --run_dir "$RUN_DIR"

echo "Done → $RUN_DIR/metrics.json"
