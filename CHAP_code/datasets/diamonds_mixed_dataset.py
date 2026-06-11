import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from datasets import IndexedReconstructDataset
import numpy as np

def load_diamonds_mixed_reconstruct_dataset(csv_path='../raw_data/diamonds_mixed.csv'):
    """
    加载 diamonds_mixed 数据集，2/3/4列为分类，其余为连续，最后一列为回归目标。
    返回：dataset, v, num_classes_dict
    """
    # 读取数据，去掉表头
    df = pd.read_csv(csv_path, header=0)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)  # 打乱
    num_features = len(df.columns)
    cat_indices = [1, 2, 3]
    cont_indices = [i for i in range(num_features) if i not in cat_indices]
    # 连续特征标准化
    scaler = StandardScaler()
    df.iloc[:, cont_indices] = scaler.fit_transform(df.iloc[:, cont_indices])
    # 分类特征编码
    label_encoders = {}
    for col_idx in cat_indices:
        le = LabelEncoder()
        df.iloc[:, col_idx] = le.fit_transform(df.iloc[:, col_idx])
        label_encoders[col_idx] = le
    # 构造v向量
    v = np.zeros(num_features, dtype=np.int64)
    v[cat_indices] = 1
    # 构造num_classes_dict
    num_classes_dict = {col_idx: len(label_encoders[col_idx].classes_) for col_idx in cat_indices}
    dataset = IndexedReconstructDataset(df, cont_indices=cont_indices, cat_indices=cat_indices)
    return dataset, v, num_classes_dict 