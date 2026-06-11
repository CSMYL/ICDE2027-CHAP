import pandas as pd
from datasets import IndexedReconstructDataset
import numpy as np
import os


def load_crime_reconstruct_dataset(csv_path=None):
    if csv_path is None:
        # 获取当前文件所在目录，然后找到项目根目录
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir)
        csv_path = os.path.join(project_root, 'raw_data', 'crime.csv')
    
    # 读取数据（有表头）
    df = pd.read_csv(csv_path, header=0)
    
    # 确保目标变量在最后一列
    target_col = 'ViolentCrimesPerPop'
    if df.columns[-1] != target_col:
        # 将目标变量移动到最后一列
        cols = [col for col in df.columns if col != target_col] + [target_col]
        df = df[cols]
    
    num_features = len(df.columns)  # 包括目标变量
    
    # 根据crime_info.json，前119个特征是连续特征，后3个是分类特征，最后1个是目标变量（连续）
    # 所以总共122列：119个连续特征 + 3个分类特征 + 1个目标变量（连续）
    cont_indices = []
    cat_indices = []
    
    for col_idx in range(num_features):
        col_name = df.columns[col_idx]
        if col_name == target_col:
            # 目标变量是连续特征（回归任务）
            cont_indices.append(col_idx)
        elif col_idx >= num_features - 4 and col_idx < num_features - 1:  # 倒数第2-4列是分类特征
            cat_indices.append(col_idx)
        else:
            # 前119个是连续特征
            cont_indices.append(col_idx)
    
    # 构造 v 向量：0表示连续特征，1表示分类特征（包括目标变量，目标变量是连续特征，所以v[target_idx]=0）
    v = np.zeros(num_features, dtype=np.int64)
    for cat_idx in cat_indices:
        v[cat_idx] = 1
    
    # 构造 num_classes_dict（分类特征的类别数）
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

