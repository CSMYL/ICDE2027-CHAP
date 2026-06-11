#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
为TabGNN创建db_info_fz.json文件
"""

import pandas as pd
import json
import os
import numpy as np

# 数据集配置
DATASETS = {
    'crime': {
        'ds_info_path': 'data/test_data/crime.ds_info.json',
        'data_path': 'data/raw_data/crime.csv',
        'task_type': 'regression',
        'n_classes': 1
    },
    'meps': {
        'ds_info_path': 'data/test_data/meps.ds_info.json',
        'data_path': 'data/raw_data/meps.csv',
        'task_type': 'regression',
        'n_classes': 1
    }
}

def create_db_info_fz(dataset_name, config):
    """创建db_info_fz.json文件"""
    print(f"\n处理数据集: {dataset_name}")
    
    # 读取ds_info.json
    with open(config['ds_info_path'], 'r', encoding='utf-8') as f:
        ds_info = json.load(f)
    
    # 读取数据
    df = pd.read_csv(config['data_path'], header=None)
    total = len(df)
    n_train = int(total * 0.8)
    n_test = total - n_train
    
    # 计算训练集的类别分布（对于回归任务，这个不太重要，但需要提供）
    train_data = df.iloc[:n_train]
    if config['task_type'] == 'regression':
        train_class_counts = []  # 回归任务为空列表
    else:
        # 分类任务：计算每个类别的数量
        target_col = len(ds_info['columns']) - 1
        train_targets = train_data.iloc[:, target_col]
        unique, counts = np.unique(train_targets, return_counts=True)
        train_class_counts = counts.tolist()
    
    # 构建db_info_fz.json
    db_info = {
        'task': {
            'type': config['task_type'],
            'n_classes': config['n_classes'],
            'n_train': n_train,
            'n_test': n_test,
            'train_class_counts': train_class_counts
        },
        'node_type_to_int': {
            'Main_table': 0
        },
        'edge_type_to_int': {
            'SELF': 0,
            'SIMILARITY_EDGE': 1
        },
        'node_types_and_features': {
            'Main_table': {}
        },
        'label_feature': 'Main_table.TARGET'
    }
    
    # 添加特征定义
    node_features = db_info['node_types_and_features']['Main_table']
    node_features['INDEX_ID'] = {'type': 'SCALAR'}
    
    # 添加所有特征
    for col_info in ds_info['columns']:
        if col_info['name'] == 'TARGET':
            if config['task_type'] == 'regression':
                node_features['TARGET'] = {'type': 'NUMERIC'}
            else:
                node_features['TARGET'] = {'type': 'CATEGORICAL', 'cardinality': col_info.get('cardinality', 2)}
        else:
            if col_info['type'] == 'NUMERIC':
                node_features[col_info['name']] = {'type': 'NUMERIC'}
            else:
                node_features[col_info['name']] = {'type': 'CATEGORICAL', 'cardinality': col_info['cardinality']}
    
    # 添加连接键（分类特征可以作为连接键）
    cat_features = [col['name'] for col in ds_info['columns'][:-1] if col['type'] == 'CATEGORICAL']
    edge_type_idx = 2
    for cat_feat in cat_features:
        db_info['edge_type_to_int'][cat_feat] = edge_type_idx
        edge_type_idx += 1
    
    # 保存文件
    db_info_path = f'data/{dataset_name}/{dataset_name}.db_info_fz.json'
    os.makedirs(os.path.dirname(db_info_path), exist_ok=True)
    with open(db_info_path, 'w', encoding='utf-8') as f:
        json.dump(db_info, f, indent=2, ensure_ascii=False)
    
    print(f"db_info_fz.json保存到: {db_info_path}")
    print(f"训练集: {n_train} 行, 测试集: {n_test} 行")
    print(f"特征数: {len(ds_info['columns']) - 1}")
    print(f"连接键数: {len(cat_features)}")
    
    return db_info

if __name__ == '__main__':
    print("=" * 60)
    print("为TabGNN创建db_info_fz.json文件")
    print("=" * 60)
    
    results = {}
    for dataset_name, config in DATASETS.items():
        if not os.path.exists(config['ds_info_path']):
            print(f"\n警告: ds_info.json不存在: {config['ds_info_path']}")
            continue
        if not os.path.exists(config['data_path']):
            print(f"\n警告: 数据文件不存在: {config['data_path']}")
            continue
        try:
            db_info = create_db_info_fz(dataset_name, config)
            results[dataset_name] = 'success'
        except Exception as e:
            print(f"\n错误处理 {dataset_name}: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("db_info_fz.json创建完成！")
    print("=" * 60)

