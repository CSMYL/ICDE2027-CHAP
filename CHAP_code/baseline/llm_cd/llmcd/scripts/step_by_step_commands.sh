#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-baseline/llmcd/configs/synthetic_demo.json}"
RUN_NAME="$(.venv/bin/python -c 'import json,sys; print(json.load(open(sys.argv[1])).get("run_name","llmcd_run"))' "$CONFIG")"
RUN_DIR="baseline/llmcd/runs/${RUN_NAME}"

cat <<EOF
################################################################################
# LLM-CD 分步命令说明
################################################################################
# 配置文件: $CONFIG
# 本次 run 目录: $RUN_DIR
#
# 产物说明:
#   1. pc_graph.json     : 第一步输出，PC 算法发现的初始图
#   2. final_graph.json  : 第二步输出，LLM 或 fallback 规则修正后的最终图
#   3. metrics.json      : 第三步输出，只用目标父节点训练/测试的结果
################################################################################

################################################################################
# 0. 可选：如果你在有网机器上调用大模型 API，先设置这些环境变量
#    注意：不要把真实 key 写进代码或配置文件。
################################################################################
export LLMCD_API_KEY="replace-with-your-api-key"
export LLMCD_BASE_URL="https://api.deepseek.com"
export LLMCD_MODEL="deepseek-v4-flash"
export MPLCONFIGDIR="\$PWD/.cache/matplotlib"
mkdir -p "$RUN_DIR" .cache/matplotlib

################################################################################
# 1. 无网服务器可执行：输入数据 -> PC 初始图
#
# 输入:
#   - $CONFIG
#   - 配置中指定的数据，或项目内置 CHAP loader
#
# 输出:
#   - $RUN_DIR/pc_graph.json
#
# 做完这一步后，如果服务器无网，需要把下面两个文件复制到有网机器:
#   - $CONFIG
#   - $RUN_DIR/pc_graph.json
################################################################################
.venv/bin/python baseline/llmcd/discover_pc_graph.py \\
  --config "$CONFIG" \\
  --run_dir "$RUN_DIR"

################################################################################
# 2A. 有网机器执行：PC 初始图 -> LLM 判断后的最终图
#
# 前提:
#   - 已经设置 LLMCD_API_KEY / LLMCD_BASE_URL / LLMCD_MODEL
#
# 输入:
#   - $CONFIG
#   - $RUN_DIR/pc_graph.json
#
# 输出:
#   - $RUN_DIR/final_graph.json
#
# 做完这一步后，把 final_graph.json 复制回无网服务器即可训练。
################################################################################
.venv/bin/python baseline/llmcd/judge_llm_graph.py \\
  --config "$CONFIG" \\
  --pc_graph "$RUN_DIR/pc_graph.json" \\
  --run_dir "$RUN_DIR" \\
  --use_llm

################################################################################
# 2B. 无网 fallback：PC 初始图 -> 规则定向后的最终图，不调用 API
#
# 如果暂时没有 API，或者只想先跑通流程，用这一段替代 2A。
#
# 输入:
#   - $CONFIG
#   - $RUN_DIR/pc_graph.json
#
# 输出:
#   - $RUN_DIR/final_graph.json
################################################################################
.venv/bin/python baseline/llmcd/judge_llm_graph.py \\
  --config "$CONFIG" \\
  --pc_graph "$RUN_DIR/pc_graph.json" \\
  --run_dir "$RUN_DIR"

################################################################################
# 3. 无网服务器可执行：最终图 -> 取目标父节点训练和测试
#
# 输入:
#   - $CONFIG
#   - $RUN_DIR/final_graph.json
#
# 输出:
#   - $RUN_DIR/metrics.json
################################################################################
.venv/bin/python baseline/llmcd/train_eval_from_graph.py \\
  --config "$CONFIG" \\
  --graph "$RUN_DIR/final_graph.json" \\
  --run_dir "$RUN_DIR"
EOF
