#!/usr/bin/env python
import os
import sys
import argparse
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from sklearn.metrics import roc_auc_score, mean_squared_error

# 添加项目路径
sys.path.insert(0, os.path.dirname(__file__))

from data.CSVToGraphAdapter import CSVToGraphAdapter
from models.GNN.GCN import GCN
from utils import get_DGL_collator


def parse_args():
    parser = argparse.ArgumentParser(description='训练GNN模型')
    
    # 数据集参数
    parser.add_argument('--dataset', type=str, required=True,
                       help='数据集名称（需在data/tabular_ds_info.json中注册）')
    parser.add_argument('--train_split', type=float, default=0.8,
                       help='训练集比例（默认0.8）')
    parser.add_argument('--connect_keys', type=str, nargs='*', default=None,
                       help='连接键列表（分类特征名），None表示自动检测或使用相似性')
    parser.add_argument('--max_neighbors', type=int, default=10,
                       help='每个连接键的最大邻居数（默认10）')
    
    # 模型参数
    parser.add_argument('--model', type=str, default='GCN',
                       choices=['GCN'],
                       help='模型类型（默认GCN）')
    parser.add_argument('--hidden_dim', type=int, default=64,
                       help='隐藏层维度（默认64）')
    parser.add_argument('--n_layers', type=int, default=3,
                       help='GNN层数（默认3）')
    parser.add_argument('--fcout_layers', type=int, nargs='+', default=[64, 32],
                       help='输出层MLP层大小（默认[64, 32]）')
    parser.add_argument('--dropout', type=float, default=0.2,
                       help='Dropout比例（默认0.2）')
    
    # 训练参数
    parser.add_argument('--epochs', type=int, default=10,
                       help='训练轮数（默认10）')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='批次大小（默认32）')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='学习率（默认0.001）')
    parser.add_argument('--device', type=str, default='cpu',
                       help='设备（默认cpu）')
    
    return parser.parse_args()


def get_metric_name(task_type):
    """根据任务类型返回评估指标名称"""
    if task_type == 'binary classification' or task_type == 'multiclass classification':
        return 'AUC'
    elif task_type == 'regression':
        return 'RMSE'
    else:
        raise ValueError(f'不支持的任务类型: {task_type}')


def evaluate(model, loader, criterion, task_type, device):
    """评估模型"""
    model.eval()
    all_labels = []
    all_outputs = []
    total_loss = 0.0
    
    with torch.no_grad():
        for batch in loader:
            (bdgl, features, main_node_ids), labels = batch
            
            # 如果图已经在 GPU 上（collator 中创建），则不需要再次迁移
            if bdgl.device != device:
            bdgl = bdgl.to(device)
            labels = labels.to(device)
            
            # 处理features（如果特征已经在 GPU 上，则不需要再次迁移）
            device_features = {}
            for node_type, node_features in features.items():
                if isinstance(node_features, tuple):
                    cat_feats, cont_feats = node_features
                    # 检查特征是否已经在正确的设备上
                    if isinstance(cat_feats, torch.Tensor) and cat_feats.device != device:
                        cat_feats = cat_feats.to(device)
                    if isinstance(cont_feats, torch.Tensor) and cont_feats.device != device:
                        cont_feats = cont_feats.to(device)
                    device_features[node_type] = (cat_feats, cont_feats)
                else:
                    device_features[node_type] = node_features
            
            input_data = (bdgl, device_features, main_node_ids)
            outputs = model(input_data)
            
            labels = labels.long()
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            
            all_labels.append(labels.cpu().numpy())
            all_outputs.append(outputs.cpu().numpy())
    
    all_labels = np.concatenate(all_labels)
    all_outputs = np.concatenate(all_outputs)
    avg_loss = total_loss / len(loader)
    
    # 计算评估指标
    if task_type == 'binary classification':
        probs = torch.softmax(torch.tensor(all_outputs), dim=1).numpy()
        metric_value = roc_auc_score(all_labels, probs[:, 1])
    elif task_type == 'multiclass classification':
        probs = torch.softmax(torch.tensor(all_outputs), dim=1).numpy()
        metric_value = roc_auc_score(all_labels, probs, multi_class='ovr')
    elif task_type == 'regression':
        predictions = all_outputs.squeeze()
        metric_value = np.sqrt(mean_squared_error(all_labels, predictions))
    else:
        raise ValueError(f'不支持的任务类型: {task_type}')
    
    return avg_loss, metric_value


def train_epoch(model, loader, criterion, optimizer, device, task_type='binary classification'):
    """训练一个epoch"""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    batch_idx = 0
    for batch in loader:
        batch_idx += 1
        
        (bdgl, features, main_node_ids), labels = batch
        
        optimizer.zero_grad()
        
        # 如果图已经在 GPU 上（collator 中创建），则不需要再次迁移
        if bdgl.device != device:
        bdgl = bdgl.to(device)
        labels = labels.to(device)
        
        # 处理features（如果特征已经在 GPU 上，则不需要再次迁移）
        device_features = {}
        for node_type, node_features in features.items():
            if isinstance(node_features, tuple):
                cat_feats, cont_feats = node_features
                # 检查特征是否已经在正确的设备上
                if isinstance(cat_feats, torch.Tensor) and cat_feats.device != device:
                    cat_feats = cat_feats.to(device)
                if isinstance(cont_feats, torch.Tensor) and cont_feats.device != device:
                    cont_feats = cont_feats.to(device)
                device_features[node_type] = (cat_feats, cont_feats)
            else:
                device_features[node_type] = node_features
        
        input_data = (bdgl, device_features, main_node_ids)
        outputs = model(input_data)
        
        # 回归任务使用float，分类任务使用long
        if task_type == 'regression':
            labels = labels.float()
            # 对于回归任务，outputs可能是(batch_size,)形状，需要squeeze
            if outputs.dim() > 1:
                outputs = outputs.squeeze()
        else:
        labels = labels.long()
        
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        if task_type == 'regression':
            # 回归任务不使用准确率
            total += labels.size(0)
        else:
            # 分类任务：计算准确率
            if outputs.dim() > 1:
        predictions = outputs.argmax(dim=1)
            else:
                # 如果outputs是1D，可能是batch_size=1的情况
                predictions = (outputs > 0.5).long()
        correct += (predictions == labels).sum().item()
        total += labels.size(0)
        
        # 每20个batch输出一次中间结果
        if batch_idx % 20 == 0:
            avg_loss = total_loss / batch_idx
            acc = correct / total if total > 0 else 0
            print(f'  Batch {batch_idx}/{len(loader)}: Loss={avg_loss:.4f}, Acc={acc:.4f}', flush=True)
    
    return total_loss / len(loader), correct / total if total > 0 else 0


def main():
    args = parse_args()
    
    print('=' * 60)
    print('GNN模型训练')
    print('=' * 60)
    print(f'数据集: {args.dataset}')
    print(f'模型: {args.model}')
    print(f'训练轮数: {args.epochs}')
    print(f'批次大小: {args.batch_size}')
    print(f'隐藏层维度: {args.hidden_dim}')
    print(f'GNN层数: {args.n_layers}')
    print('=' * 60)
    
    # 加载数据集信息
    from data.utils import get_ds_info
    try:
        ds_info = get_ds_info(args.dataset)
        # task_type 可能在 processed 或 meta 中
        if 'processed' in ds_info and 'task' in ds_info['processed']:
            task_type = ds_info['processed']['task']
        elif 'meta' in ds_info and 'task' in ds_info['meta']:
            task_type = ds_info['meta']['task']
        else:
            raise KeyError('无法找到task字段')
        print(f'\n任务类型: {task_type}')
    except Exception as e:
        print(f'✗ 无法加载数据集信息: {e}')
        import traceback
        traceback.print_exc()
        return
    
    # 编码器配置
    encoders = {
        'NUMERIC': 'ScalarRobustScalerEnc',
        'CATEGORICAL': 'CategoricalOrdinalEnc'
    }
    
    # 加载完整数据集
    print(f'\n1. 加载数据集: {args.dataset}')
    try:
        full_dataset = CSVToGraphAdapter(
            dataset_name=args.dataset,
            encoders=encoders,
            connect_keys=args.connect_keys,
            max_neighbors_per_key=args.max_neighbors
        )
        print(f'   ✓ 数据集加载成功，共 {len(full_dataset)} 个样本')
        print(f'   ✓ 连接键: {full_dataset.connect_keys}')
    except Exception as e:
        print(f'   ✗ 数据集加载失败: {e}')
        import traceback
        traceback.print_exc()
        return
    
    # 划分训练集和测试集
    n_total = len(full_dataset)
    n_train = int(n_total * args.train_split)
    train_ids = np.arange(n_train)
    test_ids = np.arange(n_train, n_total)
    
    train_dataset = CSVToGraphAdapter(
        dataset_name=args.dataset,
        datapoint_ids=train_ids,
        encoders=encoders,
        connect_keys=args.connect_keys,
        max_neighbors_per_key=args.max_neighbors
    )
    test_dataset = CSVToGraphAdapter(
        dataset_name=args.dataset,
        datapoint_ids=test_ids,
        encoders=encoders,
        connect_keys=args.connect_keys,
        max_neighbors_per_key=args.max_neighbors
    )
    
    print(f'   ✓ 训练集: {len(train_dataset)} 个样本')
    print(f'   ✓ 测试集: {len(test_dataset)} 个样本')
    
    # 设置设备（需要在创建 collator 之前）
    device = torch.device(args.device)
    
    # 创建collator
    print('\n2. 创建数据加载器')
    try:
        print('   [DEBUG] 创建collator...', flush=True)
        # 如果使用 GPU，在 collator 中直接创建 GPU 图，避免 CPU->GPU 传输
        collator_device = device if device.type == 'cuda' else 'cpu'
        collator = get_DGL_collator(train_dataset.feature_encoders, train_dataset.db_info, device=collator_device)
        print(f'   [DEBUG] Collator创建成功（device={collator_device}），创建DataLoader...', flush=True)
        # 如果数据已经在 GPU 上（collator 中创建），则不需要 pin_memory（pin_memory 只用于 CPU tensor）
        # 否则，启用 pin_memory 加速 CPU->GPU 数据传输
        pin_memory = (device.type == 'cuda' and collator_device == 'cpu')
        print(f'   [DEBUG] pin_memory={pin_memory} (collator_device={collator_device}, device={device})', flush=True)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collator, num_workers=0, pin_memory=pin_memory)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collator, num_workers=0, pin_memory=pin_memory)
        print(f'   ✓ 数据加载器创建成功，训练集batch数: {len(train_loader)}, 测试集batch数: {len(test_loader)}', flush=True)
    except Exception as e:
        print(f'   ✗ 数据加载器创建失败: {e}', flush=True)
        import traceback
        traceback.print_exc()
        return
    
    # 初始化模型
    print('\n3. 初始化模型')
    try:
        if args.model == 'GCN':
            model = GCN(
                writer=None,
                dataset_name=args.dataset,
                feature_encoders=train_dataset.feature_encoders,
                hidden_dim=args.hidden_dim,
                init_model_class_name='TabMLP',
                init_model_kwargs={
                    'layer_sizes': [args.hidden_dim],
                    'max_emb_dim': 50,
                    'activation_class_name': 'GELU',
                    'activation_class_kwargs': {},
                    'norm_class_name': 'BatchNorm1d',
                    'norm_class_kwargs': {},
                    'one_hot_embeddings': False,
                    'drop_whole_embeddings': False,
                    'p_dropout': args.dropout * 0.5
                },
                n_layers=args.n_layers,
                activation_class_name='GELU',
                activation_class_kwargs={},
                norm_class_name='BatchNorm1d',
                norm_class_kwargs={},
                loss_class_name='CrossEntropyLoss' if task_type != 'regression' else 'MSELoss',
                loss_class_kwargs={},
                p_dropout=args.dropout,
                readout_class_name='GlobalAttentionPooling',
                readout_kwargs={
                    'n_layers': 2,
                    'act_name': 'GELU'
                },
                fcout_layer_sizes=args.fcout_layers,
                use_jknet=False,
                cat_fz_embedding=False
            )
        else:
            raise ValueError(f'不支持的模型: {args.model}')
        
        n_params = sum(p.numel() for p in model.parameters())
        print(f'   ✓ 模型初始化成功')
        print(f'   ✓ 模型参数数量: {n_params:,}')
    except Exception as e:
        print(f'   ✗ 模型初始化失败: {e}')
        import traceback
        traceback.print_exc()
        return
    
    # 设置设备和优化器
    device = torch.device(args.device)
    model.to(device)
    
    if task_type == 'regression':
        criterion = nn.MSELoss()
    else:
        criterion = nn.CrossEntropyLoss()
    
    optimizer = Adam(model.parameters(), lr=args.lr)
    
    # 训练循环
    print(f'\n4. 开始训练（{args.epochs}个epoch）')
    metric_name = get_metric_name(task_type)
    
    best_test_metric = float('inf') if task_type == 'regression' else 0.0
    
    for epoch in range(1, args.epochs + 1):
        # 训练
        train_start_time = time.time()
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, task_type)
        train_time = time.time() - train_start_time
        train_time_min = train_time / 60
        
        # 测试
        test_start_time = time.time()
        test_loss, test_metric = evaluate(model, test_loader, criterion, task_type, device)
        test_time = time.time() - test_start_time
        test_time_min = test_time / 60
        # 判断是否最佳
        is_best = False
        if task_type == 'regression':
            if test_metric < best_test_metric:
                best_test_metric = test_metric
                is_best = True
        else:
            if test_metric > best_test_metric:
                best_test_metric = test_metric
                is_best = True
        
        # 输出结果
        marker = ' *' if is_best else ''
        if task_type == 'regression':
            print(f'Epoch {epoch}/{args.epochs}: '
                  f'Train Loss: {train_loss:.4f} ({train_time_min:.2f}min) | '
                  f'Test Loss: {test_loss:.4f}, Test {metric_name}: {test_metric:.4f} ({test_time_min:.2f}min){marker}')
        else:
            print(f'Epoch {epoch}/{args.epochs}: '
                  f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} ({train_time_min:.2f}min) | '
                  f'Test Loss: {test_loss:.4f}, Test {metric_name}: {test_metric:.4f} ({test_time_min:.2f}min){marker}')
    
    print('\n' + '=' * 60)
    print('✓ 训练完成！')
    print(f'✓ 最佳测试{metric_name}: {best_test_metric:.4f}')
    print('=' * 60)


if __name__ == '__main__':
    main()

