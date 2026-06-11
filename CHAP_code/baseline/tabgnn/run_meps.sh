#!/bin/bash
# TabGNN GPU训练运行脚本
# 使用conda环境 tabgnn_clean
# 超参数配置在脚本中，可直接修改

cd "$(dirname "$0")"

# 激活conda环境
source ~/miniconda3/etc/profile.d/conda.sh
conda activate tabgnn_clean

# 设置CUDA库路径
export LD_LIBRARY_PATH="/usr/local/cuda-12.2/targets/x86_64-linux/lib:/usr/local/cuda-12.2/lib64:$LD_LIBRARY_PATH"

# ============================================
# 超参数配置（可直接修改此部分）
# ============================================
DATASET="meps"              # 数据集名称
EPOCHS=50                    # 训练轮数（housesale样本较少，增加轮数以充分训练）
BATCH_SIZE=256                # 批次大小（保持64，样本数适中）
HIDDEN_DIM=128               # 隐藏层维度（保持128，利用分类特征连接的优势）
N_LAYERS=3                   # GNN层数（保持3层，分类特征连接通常不需要太深）
DROPOUT=0.25                 # Dropout比例（housesale有分类特征，过拟合风险较低，降低到0.25）
LR=1e-5                    # 学习率（分类特征连接更稳定，使用中等学习率）
TRAIN_SPLIT=0.8              # 训练集比例
DEVICE="cuda:1"                # 设备 (cuda 或 cpu)
# FCOUT_LAYERS="64 32"       # 输出层MLP层大小（默认[64, 32]）
# CONNECT_KEYS=""            # 连接键列表，留空表示自动检测
# MAX_NEIGHBORS=10           # 每个连接键的最大邻居数
# ============================================

# 构建训练命令
CMD="python train.py \
    --dataset $DATASET \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --hidden_dim $HIDDEN_DIM \
    --n_layers $N_LAYERS \
    --dropout $DROPOUT \
    --lr $LR \
    --train_split $TRAIN_SPLIT \
    --device $DEVICE"

# 如果命令行有参数，则使用命令行参数（覆盖脚本中的配置）
if [ $# -gt 0 ]; then
    CMD="python train.py $@"
fi

# 运行训练
echo "=========================================="
echo "TabGNN GPU训练"
echo "=========================================="
echo "数据集: $DATASET"
echo "训练轮数: $EPOCHS"
echo "批次大小: $BATCH_SIZE"
echo "隐藏层维度: $HIDDEN_DIM"
echo "GNN层数: $N_LAYERS"
echo "设备: $DEVICE"
echo "=========================================="
echo ""

$CMD
