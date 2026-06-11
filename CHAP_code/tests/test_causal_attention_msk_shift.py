#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
基于 Diamonds OOD shift1/shift2/shift3 的 Causal Attention 实验脚本。

与原始 `test_causal_attention_msk_model.py` 的主要区别：
- 不再在代码里按 8:2 重新划分数据集；
- 而是尊重我们事先构造好的 OOD 划分：
  - 对于 shiftX：前 N_train 行来自 train_expX（OOD Train），后 N_test 行来自 test_expX（OOD Test）
- 仍然会从 OOD Train 部分再切一块出来作为 Val（例如 80/20 随机拆分），但不会动 OOD Test。
"""

import argparse
import logging
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, random_split

from datasets.ood_shift_datasets import _load_shift_reconstruct_dataset
from models.mask_generators import SigmoidMaskGenerator


def get_model_class(model_source="models.causal_attention_msk_model"):
    """与 test_causal_attention_msk_model.py 一致：根据 model_source 动态导入模型类。"""
    try:
        module = __import__(model_source, fromlist=["CausalAttentionMskModel"])
        return getattr(module, "CausalAttentionMskModel")
    except ImportError as e:
        raise ImportError(f"无法导入模型模块 {model_source}: {e}")
    except AttributeError as e:
        raise AttributeError(f"模块 {model_source} 中没有找到 CausalAttentionMskModel 类: {e}")


def get_train_function(train_source="utils.train_msk_utils"):
    """与 test_causal_attention_msk_model.py 一致：根据 train_source 动态导入训练函数。"""
    try:
        module = __import__(train_source, fromlist=["train_msk_model"])
        return getattr(module, "train_msk_model")
    except ImportError as e:
        raise ImportError(f"无法导入训练模块 {train_source}: {e}")
    except AttributeError as e:
        raise AttributeError(f"模块 {train_source} 中没有找到 train_msk_model 函数: {e}")


def build_shift_dataset(shift_name: str):
    """
    读取 shift1/2/3 与对应的 train_expX/test_expX 大小信息，
    返回：
        dataset_obj: 整个 shift 数据（train+test 拼接）
        v: 特征类型向量
        num_classes_dict: 分类特征类别数
        train_size: OOD Train 样本数（用于索引）
    """
    assert shift_name in ("shift1", "shift2", "shift3")
    exp_id = int(shift_name[-1])

    base_dir = os.path.join(os.path.dirname(__file__), "../raw_data/diamonds_ood_experiments")
    base_dir = os.path.abspath(base_dir)

    shift_csv = os.path.join(base_dir, f"{shift_name}.csv")
    train_csv = os.path.join(base_dir, f"train_exp{exp_id}.csv")
    test_csv = os.path.join(base_dir, f"test_exp{exp_id}.csv")

    if not os.path.exists(shift_csv):
        raise FileNotFoundError(f"{shift_csv} 不存在，请先在 raw_data 下运行 generate_ood_splits.py 和 generate_shift_datasets.py")
    if not (os.path.exists(train_csv) and os.path.exists(test_csv)):
        raise FileNotFoundError(f"缺少 OOD 划分文件: {train_csv} 或 {test_csv}")

    # 读取 shift（已标准化连续特征）；不 shuffle，保持前 N_train 行=OOD train、后 N_test 行=OOD test
    dataset_obj, v, num_classes_dict = _load_shift_reconstruct_dataset(shift_csv, shuffle=False)

    # 仅用 train_expX 的行数推断 OOD train/test 边界
    import pandas as pd

    n_train = len(pd.read_csv(train_csv))
    n_total = len(dataset_obj)
    if n_total != 2 * n_train:
        raise ValueError(
            f"{shift_name}: 期望 shift 中样本数是 2 * |train_exp{exp_id}|，"
            f"但现在 shift={n_total}, train_exp={n_train}"
        )

    return dataset_obj, v, num_classes_dict, n_train


def main(args: argparse.Namespace):
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    device = torch.device(f"cuda:{args.gpu_id}" if args.gpu_id >= 0 and torch.cuda.is_available() else "cpu")
    logger.info(f"使用设备: {device}")

    # -------- 加载 shift 数据集（整个 shift1/2/3） --------
    dataset_obj, v, num_classes_dict, ood_train_size = build_shift_dataset(args.dataset)
    total_size = len(dataset_obj)
    ood_test_size = total_size - ood_train_size

    logger.info(f"数据集: {args.dataset}")
    logger.info(f"总样本数: {total_size} (OOD train={ood_train_size}, OOD test={ood_test_size})")
    logger.info(f"特征类型向量 v: {v}")
    logger.info(f"v 形状: {v.shape}")
    logger.info(f"分类特征类别数: {num_classes_dict}")

    # 确保 v 是 tensor
    if not isinstance(v, torch.Tensor):
        v = torch.tensor(v)
    v = v.to(device)

    # 看一个样本
    sample_x, sample_y = dataset_obj[0]
    logger.info(f"样本输入形状: {sample_x.shape}")
    logger.info(f"样本目标形状: {sample_y.shape}")

    target_idx = len(v) - 1
    logger.info(f"预测目标索引（最后一列）: {target_idx}")

    # -------- 使用 OOD Train 部分再拆成 train/val，保持 OOD Test 不变 --------
    # OOD Train 的索引范围 [0, ood_train_size)
    train_val_indices = list(range(0, ood_train_size))
    test_indices = list(range(ood_train_size, total_size))

    ood_train_dataset = Subset(dataset_obj, train_val_indices)
    test_dataset = Subset(dataset_obj, test_indices)

    train_val_size = len(ood_train_dataset)
    val_size = int(args.val_ratio * train_val_size)
    train_size = train_val_size - val_size

    train_dataset, val_dataset = random_split(
        ood_train_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    logger.info(f"O O D Train 总大小: {train_val_size}")
    logger.info(f"  → 训练集: {len(train_dataset)} ({len(train_dataset)/total_size*100:.1f}%)")
    logger.info(f"  → 验证集: {len(val_dataset)} ({len(val_dataset)/total_size*100:.1f}%)")
    logger.info(f"O O D Test 大小: {len(test_dataset)} ({len(test_dataset)/total_size*100:.1f}%)")

    # DataLoader
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # -------- Mask generator、模型和训练 --------
    logger.info("\n初始化 mask 生成器...")
    mask_generator = SigmoidMaskGenerator(
        num_features=len(v),
        initial_threshold=0.2,
        final_threshold=0.2,
        threshold_multiplier=1.1,
    ).to(device)

    logger.info(f"\n从 {args.model_source} 导入模型类...")
    ModelClass = get_model_class(args.model_source)

    logger.info(f"\n从 {args.train_source} 导入训练函数...")
    train_function = get_train_function(args.train_source)

    logger.info("\n初始化因果注意力 MSK 模型...")
    model = ModelClass(
        v=v,
        num_classes_dict=num_classes_dict,
        d_model=args.d_model,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        dropout=args.dropout,
        share_embedding=True,
        mask_generator=mask_generator,
        target_idx=target_idx,
    ).to(device)

    logger.info(f"模型参数数量: {sum(p.numel() for p in model.parameters()):,}")

    logger.info("\n开始训练模型 (基于 shift OOD 划分)...")
    trained_model = train_function(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        v=v,
        num_classes_dict=num_classes_dict,
        test_loader=test_loader,
        prediction_idx=target_idx,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        num_epochs=args.num_epochs,
        patience=args.patience,
        alpha=args.alpha,
        beta=args.beta,
        regression_weight=args.regression_weight,
        recon_update_strategy=tuple(args.recon_update_strategy),
        pred_update_strategy=tuple(args.pred_update_strategy),
        optimizer_name=args.optimizer,
        scheduler_name=args.scheduler,
        device=device,
        save_dir=f"checkpoints_causal_msk_{args.prefix}" if args.prefix else "checkpoints_causal_msk",
        log_interval=100,
    )

    logger.info("\n开始在 OOD Test 上评估...")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Causal Attention MSK OOD Shift 实验")

    parser.add_argument(
        "--dataset",
        type=str,
        default="shift1",
        choices=["shift1", "shift2", "shift3"],
        help="选择 OOD shift 数据集：shift1(Cut), shift2(Color), shift3(Simpson)（默认：shift1）",
    )
    parser.add_argument("--prefix", type=str, default="", help="结果文件前缀")
    parser.add_argument("--gpu_id", type=int, default=-1, help="GPU 编号（-1 默认，-2 强制 CPU）")

    parser.add_argument("--d_model", type=int, default=64)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--num_layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.1)

    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.2,
        help="在 OOD Train 内部划分验证集比例（默认 0.2）",
    )
    parser.add_argument("--optimizer", type=str, default="adam")
    parser.add_argument("--scheduler", type=str, default="reduce_on_plateau")
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=0.001)
    parser.add_argument("--regression_weight", type=float, default=5.0)
    parser.add_argument(
        "--recon_update_strategy",
        type=int,
        nargs=3,
        default=[2, 10, 1],
        metavar=("NUM", "INTERVAL", "OFFSET"),
    )
    parser.add_argument(
        "--pred_update_strategy",
        type=int,
        nargs=3,
        default=[8, 10, 0],
        metavar=("NUM", "INTERVAL", "OFFSET"),
    )

    parser.add_argument(
        "--model_source",
        type=str,
        default="models.causal_attention_msk_model",
        help="模型类所在的 Python 模块路径（与主脚本一致）",
    )
    parser.add_argument(
        "--train_source",
        type=str,
        default="utils.train_msk_utils",
        help="训练函数所在的 Python 模块路径（与主脚本一致）",
    )

    args = parser.parse_args()
    main(args)

