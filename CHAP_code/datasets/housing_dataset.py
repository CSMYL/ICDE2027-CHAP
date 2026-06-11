import pandas as pd
from sklearn.preprocessing import LabelEncoder
from datasets import IndexedReconstructDataset
import numpy as np
import os

def load_housing_reconstruct_dataset(csv_path=None):
    if csv_path is None:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir)
        csv_path = os.path.join(project_root, 'raw_data', 'housing.csv')
    """
    Load housing dataset. All columns except area are categorical.
    Column 5 (area) is discretized into bins of 50. Last column is regression target.
    Returns: dataset, v, num_classes_dict
    """
    df = pd.read_csv(csv_path, header=0)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    num_features = len(df.columns)
    area_bins = (df.iloc[:, 4] // 50).astype(int)
    df.iloc[:, 4] = area_bins
    cat_indices = [i for i in range(num_features) if i != 4 and i != num_features - 1]
    cont_indices = [4, num_features - 1]
    label_encoders = {}
    for col_idx in cat_indices:
        le = LabelEncoder()
        df.iloc[:, col_idx] = le.fit_transform(df.iloc[:, col_idx])
        label_encoders[col_idx] = le
    v = np.ones(num_features, dtype=np.int64)
    v[cont_indices] = 0
    num_classes_dict = {col_idx: len(label_encoders[col_idx].classes_) for col_idx in cat_indices}
    dataset = IndexedReconstructDataset(df, cont_indices=cont_indices, cat_indices=cat_indices)
    return dataset, v, num_classes_dict
