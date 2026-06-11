import pandas as pd
import numpy as np
import os

from datasets import IndexedReconstructDataset


def _load_shift_reconstruct_dataset(csv_path: str, shuffle: bool = True):
    """
    通用 loader：加载 diamonds OOD shift 数据集（shift1/2/3）。

    假设列为：
        carat, cut, color, clarity, depth, table, x, y, z, price
    其中：
        - 连续特征: carat, depth, table, x, y, z, price （在生成 shift 时已标准化）
        - 分类特征: cut, color, clarity （已按 diamonds_mapping.csv 编码为整数）

    shuffle: 若 True 则打乱行（用于随机划分）；若 False 则保持 CSV 行序（用于 OOD：前 N_train 行=train，后 N_test 行=test）。

    返回：
        dataset: IndexedReconstructDataset
        v: (F,) 的 0/1 向量，1 表示分类特征
        num_classes_dict: {cat_idx: num_classes}
    """
    df = pd.read_csv(csv_path, header=0)
    if shuffle:
        df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)

    num_features = len(df.columns)
    # 列顺序: carat(0), cut(1), color(2), clarity(3), depth(4), table(5), x(6), y(7), z(8), price(9)
    cat_indices = [1, 2, 3]
    cont_indices = [i for i in range(num_features) if i not in cat_indices]

    # v: 1 表示分类特征，其余为 0
    v = np.zeros(num_features, dtype=np.int64)
    v[cat_indices] = 1

    # 分类特征类别数，直接用唯一值数量（值已是整数编码）
    num_classes_dict = {
        idx: int(df.iloc[:, idx].nunique())
        for idx in cat_indices
    }

    dataset = IndexedReconstructDataset(df, cont_indices=cont_indices, cat_indices=cat_indices)
    return dataset, v, num_classes_dict


def load_shift1_reconstruct_dataset(
    csv_path: str = None,
):
    """加载 shift1 数据集（基于 Cut Shift OOD 实验拼接后的完整表）"""
    if csv_path is None:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir)
        csv_path = os.path.join(project_root, ‘raw_data’, ‘diamonds_ood_experiments’, ‘shift1.csv’)
    return _load_shift_reconstruct_dataset(csv_path)


def load_shift2_reconstruct_dataset(
    csv_path: str = None,
):
    """加载 shift2 数据集（基于 Color Shift OOD 实验拼接后的完整表）"""
    if csv_path is None:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir)
        csv_path = os.path.join(project_root, ‘raw_data’, ‘diamonds_ood_experiments’, ‘shift2.csv’)
    return _load_shift_reconstruct_dataset(csv_path)


def load_shift3_reconstruct_dataset(
    csv_path: str = None,
):
    """加载 shift3 数据集（基于 Simpson’s Paradox OOD 实验拼接后的完整表）"""
    if csv_path is None:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir)
        csv_path = os.path.join(project_root, ‘raw_data’, ‘diamonds_ood_experiments’, ‘shift3.csv’)
    return _load_shift_reconstruct_dataset(csv_path)

