import pandas as pd
from sklearn.preprocessing import StandardScaler
from utils.generate_DAG_data import load_and_parse_data
from datasets import ReconstructDataset
import numpy as np


def load_synthetic_reconstruct_dataset(csv_path='../raw_data/synthetic_data.csv'):
    """
    加载 synthetic_data.csv，并构造 ReconstructDataset 和 v 向量。

    返回：
        dataset: ReconstructDataset 实例
        v: np.ndarray，长度为 F 的向量，0 表示连续特征，1 表示分类特征
        num_classes_dict: dict，键为分类特征的索引，值为该特征的类别数
    """
    # 读取和识别列
    df, cont_cols, cat_cols = load_and_parse_data(csv_path)

    # 标准化连续特征
    scaler = StandardScaler()
    df[cont_cols] = scaler.fit_transform(df[cont_cols])
    df[cat_cols] = df[cat_cols].astype(int)

    dataset = ReconstructDataset(df, cont_columns=cont_cols, cat_columns=cat_cols)

    # 按照原始数据的列顺序构造 v 向量
    v = []
    for col in df.columns:  # 使用原始列顺序
        v.append(0 if col in cont_cols else 1)
    v = np.array(v, dtype=np.int64)

    # 按照原始数据的列顺序构造 num_classes_dict
    num_classes_dict = {}
    for i, col in enumerate(df.columns):  # 使用原始列顺序
        if col in cat_cols:
            # 获取该分类特征的唯一值数量
            num_classes = df[col].nunique()
            num_classes_dict[i] = num_classes

    return dataset, v, num_classes_dict