import pandas as pd
from sklearn.preprocessing import LabelEncoder
from datasets import IndexedReconstructDataset
import numpy as np

def load_housing_reconstruct_dataset(csv_path='../raw_data/housing.csv'):
    """
    加载 housing 数据集，除area列外全为分类，第5列area按每50一档离散化，最后一列为回归目标。
    返回：dataset, v, num_classes_dict
    """
    df = pd.read_csv(csv_path, header=0)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)  # 打乱
    num_features = len(df.columns)
    # area列（第5列，下标4）离散化
    area_bins = (df.iloc[:,4] // 50).astype(int)
    df.iloc[:,4] = area_bins
    # 除area外全为分类，area和label为连续
    cat_indices = [i for i in range(num_features) if i != 4 and i != num_features-1]
    cont_indices = [4, num_features-1]
    # 分类特征编码
    label_encoders = {}
    for col_idx in cat_indices:
        le = LabelEncoder()
        df.iloc[:, col_idx] = le.fit_transform(df.iloc[:, col_idx])
        label_encoders[col_idx] = le
    # 构造v向量
    v = np.ones(num_features, dtype=np.int64)
    v[cont_indices] = 0
    # 构造num_classes_dict
    num_classes_dict = {col_idx: len(label_encoders[col_idx].classes_) for col_idx in cat_indices}
    dataset = IndexedReconstructDataset(df, cont_indices=cont_indices, cat_indices=cat_indices)
    return dataset, v, num_classes_dict 