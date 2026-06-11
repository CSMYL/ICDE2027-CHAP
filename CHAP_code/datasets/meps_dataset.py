import pandas as pd
from datasets import IndexedReconstructDataset
import numpy as np
import os


def load_meps_reconstruct_dataset(csv_path=None):
    """
    加载 meps 数据集，并构造 IndexedReconstructDataset 和 v 向量。
    meps数据集共有138个特征：
    - 4个连续特征（已标准化）：AGE, PCS42, MCS42, K6SUM42
    - 134个分类特征（已one-hot编码）
    - 最后一列是回归目标：UTILIZATION_reg
    - 注意：数据中可能包含Unnamed: 0和PERWT15F列，需要排除

    返回：
        dataset: IndexedReconstructDataset 实例
        v: np.ndarray，长度为 F 的向量，0 表示连续特征，1 表示分类特征
        num_classes_dict: dict，键为分类特征的索引，值为该特征的类别数
    """
    if csv_path is None:
        # 获取当前文件所在目录，然后找到项目根目录
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir)
        csv_path = os.path.join(project_root, 'raw_data', 'meps.csv')
    
    # 读取数据（有表头）
    df = pd.read_csv(csv_path, header=0)
    
    # 排除不需要的列（Unnamed: 0, PERWT15F等），但保留目标变量UTILIZATION_reg
    columns_to_drop = []
    for col in df.columns:
        if 'Unnamed' in col or 'PERWT' in col:
            columns_to_drop.append(col)
    
    df = df.drop(columns=columns_to_drop)
    
    # 确保目标变量在最后一列
    target_col = 'UTILIZATION_reg'
    if df.columns[-1] != target_col:
        # 将目标变量移动到最后一列
        cols = [col for col in df.columns if col != target_col] + [target_col]
        df = df[cols]
    
    num_features = len(df.columns)  # 包括目标变量
    
    # 根据meps_info.json，前4个连续特征是：AGE, PCS42, MCS42, K6SUM42
    # 目标变量UTILIZATION_reg也是连续特征（回归任务）
    continuous_feature_names = ['AGE', 'PCS42', 'MCS42', 'K6SUM42', target_col]
    cont_indices = []
    cat_indices = []
    
    for col_idx, col_name in enumerate(df.columns):
        if col_name in continuous_feature_names:
            cont_indices.append(col_idx)
        else:
            cat_indices.append(col_idx)
    
    # 构造 v 向量：0表示连续特征，1表示分类特征（目标变量是连续特征，所以v[target_idx]=0）
    v = np.zeros(num_features, dtype=np.int64)
    for cat_idx in cat_indices:
        v[cat_idx] = 1
    
    # 构造 num_classes_dict（分类特征的类别数）
    # 对于one-hot编码的特征，通常是2（二进制）
    num_classes_dict = {}
    for col_idx in cat_indices:
        unique_vals = df.iloc[:, col_idx].nunique()
        num_classes_dict[col_idx] = unique_vals
    
    # 将分类特征转换为整数（如果还不是整数的话）
    for col_idx in cat_indices:
        col_data = df.iloc[:, col_idx]
        if not pd.api.types.is_integer_dtype(col_data):
            # 如果是浮点数，先四舍五入再转换
            df.iloc[:, col_idx] = col_data.round().astype(int)
    
    dataset = IndexedReconstructDataset(df, cont_indices=cont_indices, cat_indices=cat_indices)
    
    return dataset, v, num_classes_dict

