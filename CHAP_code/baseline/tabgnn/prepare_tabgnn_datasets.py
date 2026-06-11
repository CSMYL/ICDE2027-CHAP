#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pandas as pd
import json
import os
import numpy as np

# Paths: use centralized raw_data/ at project root
_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_CURRENT_DIR))
_RAW_DATA_DIR = os.path.join(_PROJECT_ROOT, 'raw_data')

# 数据集配置
DATASETS = {
    'crime': {
        'source': os.path.join(_RAW_DATA_DIR, 'crime.csv'),
        'target_dir': os.path.dirname(os.path.abspath(__file__)),
        'task': 'regression',
        'continuous_count': 119,  # 前119个是连续特征
        'categorical_count': 3    # 后3个是分类特征
    },
    'meps': {
        'source': os.path.join(_RAW_DATA_DIR, 'meps.csv'),
        'target_dir': os.path.dirname(os.path.abspath(__file__)),
        'task': 'regression',
        'continuous_features': ['AGE', 'PCS42', 'MCS42', 'K6SUM42']  # 这4个是连续特征
    }
}

def prepare_tabgnn_dataset(dataset_name, config):
    """为TabGNN准备数据集"""
    print(f"\n处理数据集: {dataset_name}")
    print(f"源文件: {config['source']}")
    
    # 读取数据
    df = pd.read_csv(config['source'], header=0)
    
    # 对于meps，排除不需要的列
    if dataset_name == 'meps':
        cols_to_drop = [c for c in df.columns if 'Unnamed' in c or 'PERWT' in c]
        if cols_to_drop:
            df = df.drop(columns=cols_to_drop)
            print(f"已排除列: {cols_to_drop}")
    
    # 创建ds_info.json
    ds_info = {
        'task': config['task'],
        'columns': []
    }
    
    # 分离特征和标签
    if dataset_name == 'crime':
        # Crime: 前119个连续，后3个分类，最后1个标签
        feature_cols = df.columns[:-1]
        for i, col in enumerate(feature_cols):
            if i < config['continuous_count']:
                ds_info['columns'].append({'name': f'feature{i}', 'type': 'NUMERIC'})
            else:
                unique_vals = int(df[col].nunique())
                ds_info['columns'].append({'name': f'feature{i}', 'type': 'CATEGORICAL', 'cardinality': unique_vals})
    elif dataset_name == 'meps':
        # MEPS: 4个连续特征，其余分类特征
        feature_cols = df.columns[:-1]  # 排除标签列
        feature_idx = 0
        for col in feature_cols:
            if col in config['continuous_features']:
                ds_info['columns'].append({'name': f'feature{feature_idx}', 'type': 'NUMERIC'})
            else:
                unique_vals = int(df[col].nunique())
                ds_info['columns'].append({'name': f'feature{feature_idx}', 'type': 'CATEGORICAL', 'cardinality': unique_vals})
            feature_idx += 1
    
    # 添加标签列
    ds_info['columns'].append({'name': 'TARGET', 'type': 'NUMERIC'})
    
    # 保存ds_info.json
    ds_info_path = os.path.join(config['target_dir'], 'data', 'test_data', f'{dataset_name}.ds_info.json')
    os.makedirs(os.path.dirname(ds_info_path), exist_ok=True)
    with open(ds_info_path, 'w', encoding='utf-8') as f:
        json.dump(ds_info, f, indent=2, ensure_ascii=False)
    print(f"ds_info.json保存到: {ds_info_path}")
    
    # 复制数据到raw_data（重命名为feature0, feature1, ..., TARGET格式）
    raw_data_dir = os.path.join(config['target_dir'], 'data', 'raw_data')
    os.makedirs(raw_data_dir, exist_ok=True)
    
    # 重命名列
    df_renamed = df.copy()
    new_columns = []
    for col_info in ds_info['columns']:
        if col_info['name'] == 'TARGET':
            # 找到原始标签列
            original_label_col = df.columns[-1]
            new_columns.append(original_label_col)
        else:
            # 找到对应的原始列
            col_idx = int(col_info['name'].replace('feature', ''))
            if dataset_name == 'crime':
                original_col = df.columns[col_idx]
            else:  # meps
                # 需要根据连续特征列表和顺序来映射
                feature_cols = [c for c in df.columns if c not in cols_to_drop and c != df.columns[-1]]
                original_col = feature_cols[col_idx]
            new_columns.append(original_col)
    
    # 重新排列列顺序
    df_renamed = df_renamed[new_columns]
    
    # 重命名为feature0, feature1, ..., TARGET
    rename_dict = {}
    for i, col_info in enumerate(ds_info['columns']):
        rename_dict[new_columns[i]] = col_info['name']
    df_renamed = df_renamed.rename(columns=rename_dict)
    
    # 保存CSV（无表头）
    csv_path = os.path.join(raw_data_dir, f'{dataset_name}.csv')
    df_renamed.to_csv(csv_path, index=False, header=False)
    print(f"数据文件保存到: {csv_path}")
    print(f"数据形状: {df_renamed.shape}")
    
    return ds_info, df_renamed.shape

if __name__ == '__main__':
    print("=" * 60)
    print("为TabGNN准备数据集")
    print("=" * 60)
    
    results = {}
    for dataset_name, config in DATASETS.items():
        if not os.path.exists(config['source']):
            print(f"\n警告: 源文件不存在: {config['source']}")
            continue
        try:
            ds_info, shape = prepare_tabgnn_dataset(dataset_name, config)
            results[dataset_name] = {
                'shape': shape,
                'num_features': len(ds_info['columns']) - 1,
                'task': ds_info['task']
            }
        except Exception as e:
            print(f"\n错误处理 {dataset_name}: {e}")
            import traceback
            traceback.print_exc()
    
    # 打印总结
    print("\n" + "=" * 60)
    print("数据集准备完成！")
    print("=" * 60)
    for dataset_name, info in results.items():
        print(f"{dataset_name:12s}: {info['shape'][0]:6d} 行, {info['num_features']:3d} 特征, 任务: {info['task']}")
    print("=" * 60)

