import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from datasets import IndexedReconstructDataset
import numpy as np
import os

def load_diamonds_mixed_reconstruct_dataset(csv_path=None):
    if csv_path is None:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir)
        csv_path = os.path.join(project_root, 'raw_data', 'diamonds_mixed.csv')
    """
    Load diamonds_mixed dataset. Columns 2/3/4 are categorical, rest are continuous,
    last column is regression target.
    Returns: dataset, v, num_classes_dict
    """
    df = pd.read_csv(csv_path, header=0)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    num_features = len(df.columns)
    cat_indices = [1, 2, 3]
    cont_indices = [i for i in range(num_features) if i not in cat_indices]
    scaler = StandardScaler()
    df.iloc[:, cont_indices] = scaler.fit_transform(df.iloc[:, cont_indices])
    label_encoders = {}
    for col_idx in cat_indices:
        le = LabelEncoder()
        df.iloc[:, col_idx] = le.fit_transform(df.iloc[:, col_idx])
        label_encoders[col_idx] = le
    v = np.zeros(num_features, dtype=np.int64)
    v[cat_indices] = 1
    num_classes_dict = {col_idx: len(label_encoders[col_idx].classes_) for col_idx in cat_indices}
    dataset = IndexedReconstructDataset(df, cont_indices=cont_indices, cat_indices=cat_indices)
    return dataset, v, num_classes_dict
