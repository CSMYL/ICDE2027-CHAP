"""
CSV to Graph Adapter - 完全按照TabGNN论文设计实现Multiplex Graph构建

核心设计（来自TabGNN论文）：
1. **每个样本对应一个Main_table节点**（不是特征作为节点）
2. **基于样本之间的连接键（connect_key）构建多重图**：
   - 相同连接键值的样本之间有边连接
   - 不同的连接键对应不同的边类型（Multiplex Graph）
3. **每个样本的图包含**：
   - 中心节点（当前样本）
   - 邻居节点（通过连接键找到的相关样本）
   - 边：中心节点连接到邻居节点（不同的连接键=不同的边类型）
"""
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import dgl
from dgl import DGLGraph
from collections import defaultdict
import random

from __init__ import data_root
from data.TabularDataset import TabularDataset
from data.utils import get_ds_info


class CSVToGraphAdapter(Dataset):
    """
    将TabularDataset转换为多重图结构（Multiplex Graph），完全按照TabGNN论文设计
    
    核心实现（参考build_dataset_from_kaggle_files.py）：
    1. 每个样本是一个节点（Main_table类型）
    2. 基于连接键（connect_key）找到相关样本作为邻居
    3. 构建图：中心节点(0) + 邻居节点(1,2,...,N) + 边（不同类型）
    """
    
    def __init__(self, dataset_name, datapoint_ids=None, encoders=None, 
                 connect_keys=None, max_neighbors_per_key=20):
        """
        Args:
            dataset_name: 数据集名称
            datapoint_ids: 数据点ID列表（对应CSV中的行索引）
            encoders: 特征编码器配置
            connect_keys: 连接键列表（用于构建多重图的键）
                         例如：['user_id', 'category1'] 表示基于这两个特征连接样本
                         None: 自动使用所有分类特征作为连接键
            max_neighbors_per_key: 每个连接键最多找多少个邻居
        """
        self.dataset_name = dataset_name
        self.datapoint_ids = datapoint_ids
        self.connect_keys = connect_keys
        self.max_neighbors_per_key = max_neighbors_per_key
        
        # 加载表格数据
        self.tabular_dataset = TabularDataset(dataset_name=dataset_name, 
                                             datapoint_ids=datapoint_ids, 
                                             encoders=encoders)
        
        # 拟合并编码特征（用于节点特征编码）
        if self.tabular_dataset.encoders is not None:
            self.tabular_dataset.fit_feat_encoders()
            self.tabular_dataset.encode(self.tabular_dataset.feature_encoders)
        
        self.n_samples = len(self.tabular_dataset)
        
        # 获取原始数据（用于查找连接键值）
        self.raw_data = self.tabular_dataset.raw_data
        
        # 确定datapoint_ids（TabularDataset的datapoint_ids可能是None，需要处理）
        if datapoint_ids is None:
            # 使用原始数据的索引
            if hasattr(self.tabular_dataset, 'datapoint_ids') and self.tabular_dataset.datapoint_ids is not None:
                self.datapoint_ids = list(self.tabular_dataset.datapoint_ids)
            else:
                # 使用原始数据的索引
                self.datapoint_ids = list(self.raw_data.index)
        else:
            self.datapoint_ids = list(datapoint_ids)
        
        # 确定连接键（如果没有指定，使用分类特征）
        if self.connect_keys is None:
            self.connect_keys = self._auto_detect_connect_keys()
        
        # 构建图的邻接关系（预处理所有样本之间的连接）
        self._build_graph_structure()
        
        # 创建db_info格式的配置（用于GNN模型）
        self.db_info = self._create_db_info()
        
    def _auto_detect_connect_keys(self):
        """自动检测连接键：使用分类特征作为连接键"""
        connect_keys = []
        for col_info in self.tabular_dataset.columns:
            if col_info.get('type') == 'CATEGORICAL':
                col_name = col_info['name']
                if col_name in self.raw_data.columns and col_name != 'TARGET':
                    connect_keys.append(col_name)
        # 如果没找到分类特征，使用数值特征聚类作为连接键
        if len(connect_keys) == 0:
            print("  警告: 未找到分类特征作为连接键，将使用所有特征进行相似性连接")
            # 使用KNN方法
            connect_keys = ['__similarity__']
        return connect_keys
    
    def _build_graph_structure(self):
        """
        构建样本之间的多重图结构（Multiplex Graph）
        参考build_dataset_from_kaggle_files.py的实现
        """
        # 构建连接键字典（connect_key_dict_list）
        # 格式：{连接键索引: {键值: [样本ID列表]}}
        self.connect_key_dict_list = []
        
        for key_idx, key in enumerate(self.connect_keys):
            if key == '__similarity__':
                # 特殊键：基于特征相似性
                self._build_similarity_connections(key_idx)
            else:
                # 常规键：基于分类特征相同值
                key_dict = defaultdict(list)
                for idx, sample_id in enumerate(self.datapoint_ids):
                    # TabularDataset的索引：由于CSV读取时header=None，第一行是列名
                    # 实际数据从索引1开始，所以需要idx+1
                    try:
                        # 使用位置索引（跳过第一行列名）
                        data_idx = idx + 1  # 跳过第一行（列名）
                        if data_idx < len(self.raw_data):
                            key_value = self.raw_data.iloc[data_idx][key]
                        elif sample_id in self.raw_data.index:
                            # 使用原始索引（如果sample_id是实际索引）
                            key_value = self.raw_data.loc[sample_id, key]
                        else:
                            continue
                    except (KeyError, IndexError):
                        continue
                    
                    # 跳过列名行（如果key_value是列名本身）
                    if key_value == key:
                        continue
                    
                    # 处理NaN值
                    if pd.isna(key_value):
                        key_value = '__NA__'
                    key_dict[key_value].append(idx)
                self.connect_key_dict_list.append(key_dict)
    
    def _build_similarity_connections(self, key_idx):
        """基于特征相似性构建连接（用于没有分类特征的情况）"""
        from sklearn.neighbors import NearestNeighbors
        from sklearn.preprocessing import StandardScaler
        
        # 获取所有样本的特征向量
        all_features = []
        for idx in range(len(self.tabular_dataset)):
            (cat_feats, cont_feats), _ = self.tabular_dataset[idx]
            
            # 合并特征
            if cat_feats is not None and len(cat_feats) > 0:
                if isinstance(cat_feats, torch.Tensor):
                    if isinstance(cont_feats, torch.Tensor):
                        feat_vec = torch.cat([cat_feats.flatten(), cont_feats.flatten()])
                    else:
                        feat_vec = cat_feats.flatten()
                else:
                    feat_vec = torch.cat([torch.tensor(cat_feats).flatten(), cont_feats.flatten()])
            else:
                if isinstance(cont_feats, torch.Tensor):
                    feat_vec = cont_feats.flatten()
                else:
                    feat_vec = torch.tensor(cont_feats).flatten()
            
            if isinstance(feat_vec, torch.Tensor):
                all_features.append(feat_vec.numpy())
            else:
                all_features.append(feat_vec)
        
        all_features = np.array(all_features)
        
        # 标准化
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(all_features)
        
        # KNN找到每个样本的邻居
        n_neighbors = min(self.max_neighbors_per_key + 1, len(all_features))
        knn = NearestNeighbors(n_neighbors=n_neighbors, metric='cosine')
        knn.fit(features_scaled)
        distances, indices = knn.kneighbors(features_scaled)
        
        # 构建字典格式（模拟分类特征的格式）
        key_dict = defaultdict(list)
        for i, neighbors in enumerate(indices):
            # 使用相似性作为"键值"（实际上将所有相似样本作为邻居）
            key_dict['__similarity__'].extend([(i, j) for j in neighbors[1:]])  # 跳过自己
        
        self.connect_key_dict_list.append(key_dict)
    
    def _create_db_info(self):
        """创建类似DatabaseDataset的db_info结构"""
        # 获取节点类型和特征信息（基于TabularDataset）
        node_types_and_features = {
            'Main_table': {}
        }
        
        # 添加INDEX_ID特征
        node_types_and_features['Main_table']['INDEX_ID'] = {'type': 'NUMERIC'}
        
        # 添加所有其他特征
        for col_info in self.tabular_dataset.columns:
            col_name = col_info['name']
            col_type = col_info.get('type', 'NUMERIC')
            
            # 转换为db_info格式的类型
            if col_type == 'CATEGORICAL':
                db_type = 'CATEGORICAL'
            elif col_type in ['NUMERIC', 'SCALAR']:
                db_type = 'SCALAR'
            else:
                db_type = 'SCALAR'
            
            node_types_and_features['Main_table'][col_name] = {'type': db_type}
        
        return {
            'node_types_and_features': node_types_and_features,
            'node_type_to_int': {'Main_table': 0},
            'edge_type_to_int': {'self': 0},  # 自环类型为0，其他边类型从1开始
            'task': {
                'n_classes': 2 if self.tabular_dataset.ds_info['processed']['task'] == 'binary classification' else 1
            },
            'label_feature': 'Main_table.TARGET'
        }
    
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        """
        返回一个数据点，格式与DatabaseDataset完全一致
        Returns:
            (edge_list, node_types, edge_types, features, label)
        
        格式说明（参考build_dataset_from_kaggle_files.py）：
        - edge_list: [(neighbor_idx, center_idx), ...]  # 邻居连接到中心
        - node_types: [0, 0, 0, ...]  # 所有节点都是Main_table类型(0)
        - edge_types: [1, 2, ...]  # 不同的连接键对应不同的边类型
        - features: {'Main_table': {'INDEX_ID': [...], 'feature_name': [...], ...}}
        - label: 标签值
        """
        # 获取当前样本的数据
        sample_id = self.datapoint_ids[idx]
        # TabularDataset返回(input, target)，其中input是(cat_feats, cont_feats)
        input_data, label = self.tabular_dataset[idx]
        if isinstance(input_data, tuple):
            cat_feats, cont_feats = input_data
        else:
            # 如果input_data不是tuple，说明可能是字符串或其他格式
            cat_feats, cont_feats = None, None
        
        # 初始化特征字典（按照DatabaseDataset格式）
        features = {
            'Main_table': {}
        }
        for col_info in self.tabular_dataset.columns:
            col_name = col_info['name']
            if col_name != 'TARGET':
                features['Main_table'][col_name] = []
        
        # 找到当前样本的邻居节点（基于连接键）
        related_instance_list = []  # 每个连接键对应一个邻居列表
        
        for key_idx, key in enumerate(self.connect_keys):
            if key == '__similarity__':
                # 相似性连接
                related_samples = []
                if key_idx < len(self.connect_key_dict_list):
                    similarity_edges = self.connect_key_dict_list[key_idx].get('__similarity__', [])
                    for source, target in similarity_edges:
                        if source == idx:
                            related_samples.append(target)
                related_instance_list.append(related_samples[:self.max_neighbors_per_key])
            else:
                # 常规连接键：找到具有相同键值的样本
                related_samples = []
                if key_idx < len(self.connect_key_dict_list):
                    key_dict = self.connect_key_dict_list[key_idx]
                    # 获取当前样本的连接键值（跳过第一行列名）
                    try:
                        # 使用iloc获取位置索引的值（跳过第一行）
                        data_idx = idx + 1  # 跳过第一行（列名）
                        if data_idx < len(self.raw_data):
                            key_value = self.raw_data.iloc[data_idx][key]
                            # 跳过列名行
                            if key_value == key:
                                key_value = None
                        elif sample_id in self.raw_data.index:
                            # 使用原始索引
                            key_value = self.raw_data.loc[sample_id, key]
                        else:
                            key_value = None
                    except (KeyError, IndexError):
                        key_value = None
                    
                    if key_value is not None:
                        if pd.isna(key_value):
                            key_value = '__NA__'
                        candidates = key_dict.get(key_value, [])
                        # 过滤掉当前样本本身，并限制数量
                        related_samples = [i for i in candidates if i != idx][:self.max_neighbors_per_key]
                related_instance_list.append(related_samples)
        
        # 构建所有节点：中心节点 + 所有邻居节点
        all_node_indices = [idx]  # 中心节点是当前样本（索引idx）
        for related_list in related_instance_list:
            all_node_indices.extend(related_list)
        all_node_indices = list(set(all_node_indices))  # 去重
        # 确保中心节点在第一个位置
        if idx in all_node_indices:
            all_node_indices.remove(idx)
        all_node_indices = [idx] + all_node_indices
        
        # 确保至少有一个节点（中心节点）
        if len(all_node_indices) == 0:
            all_node_indices = [idx]
        
        # 构建节点ID到图索引的映射
        node_id_to_graph_idx = {node_idx: graph_idx for graph_idx, node_idx in enumerate(all_node_indices)}
        
        # 构建边列表和边类型（完全按照build_dataset_from_kaggle_files.py的格式）
        edge_list = []
        edge_types = []
        
        # 中心节点索引（在图中的索引是0）
        center_graph_idx = 0
        
        # 为每个连接键构建边
        for key_idx, related_list in enumerate(related_instance_list):
            edge_type = key_idx + 1  # 边类型从1开始（0是自环）
            for neighbor_idx in related_list:
                neighbor_graph_idx = node_id_to_graph_idx[neighbor_idx]
                # 边：邻居节点 -> 中心节点（参考原始代码）
                edge_list.append((neighbor_graph_idx, center_graph_idx))
                edge_types.append(edge_type)
        
        # 节点类型（所有节点都是Main_table类型，即0）
        node_types = [0] * len(all_node_indices)
        
        # 填充特征（为每个节点添加原始特征值）
        for graph_idx, node_idx in enumerate(all_node_indices):
            node_sample_id = self.tabular_dataset.datapoint_ids[node_idx]
            
            # 添加INDEX_ID
            if 'INDEX_ID' not in features['Main_table']:
                features['Main_table']['INDEX_ID'] = []
            features['Main_table']['INDEX_ID'].append(int(node_sample_id))
            
            # 添加其他特征
            if node_sample_id in self.raw_data.index:
                for col_info in self.tabular_dataset.columns:
                    col_name = col_info['name']
                    if col_name != 'TARGET':
                        if col_name not in features['Main_table']:
                            features['Main_table'][col_name] = []
                        value = self.raw_data.loc[node_sample_id, col_name]
                        # 处理NaN
                        if pd.isna(value):
                            value = None
                        features['Main_table'][col_name].append(value)
        
        # 返回格式与DatabaseDataset完全一致：(dp_id, (edge_list, node_types, edge_types, features, label))
        dp_id = sample_id
        return dp_id, (edge_list, node_types, edge_types, features, label)
    
    @property
    def feature_encoders(self):
        """返回特征编码器（用于GNN模型初始化）"""
        from data.data_encoders import NullEnc
        
        # 需要为DatabaseDataset格式创建特征编码器
        feature_encoders = {
            'Main_table': {}
        }
        
        # INDEX_ID不需要编码器（使用NullEnc）
        feature_encoders['Main_table']['INDEX_ID'] = NullEnc()
        
        # 使用TabularDataset的特征编码器
        if hasattr(self.tabular_dataset, 'feature_encoders') and self.tabular_dataset.feature_encoders:
            # 遍历raw_data中的列（不包括TARGET）
            for col_name in self.raw_data.columns:
                if col_name != 'TARGET':
                    if col_name in self.tabular_dataset.feature_encoders:
                        # 使用TabularDataset的编码器
                        feature_encoders['Main_table'][col_name] = self.tabular_dataset.feature_encoders[col_name]
                    else:
                        # 如果没有编码器，使用NullEnc
                        feature_encoders['Main_table'][col_name] = NullEnc()
        else:
            # 如果没有特征编码器，为所有特征创建NullEnc
            for col_name in self.raw_data.columns:
                if col_name != 'TARGET':
                    feature_encoders['Main_table'][col_name] = NullEnc()
        
        return feature_encoders
