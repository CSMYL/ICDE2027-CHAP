# TabGNN - 图神经网络表格数据训练

本项目实现了基于图神经网络（GNN）的表格数据分类和回归任务训练框架。采用TabGNN论文中的Multiplex Graph设计，将表格数据转换为图结构进行训练。

## 项目简介

本项目支持：
- 将CSV格式的表格数据自动转换为多重图（Multiplex Graph）结构
- 使用图神经网络（GCN）进行训练
- 支持分类任务（AUC评估）和回归任务（RMSE评估）
- 支持基于分类特征的连接键或基于KNN相似性的连接键

## 支持的数据集

当前已适配的数据集：

| 数据集 | 任务类型 | 评估指标 | 特征类型 | 数据集大小 |
|--------|---------|---------|---------|-----------|
| adult | 二分类 | AUC | 全部连续 | 48,842 |
| cardio | 二分类 | AUC | 全部分类 | 70,000 |
| creditcard | 二分类 | AUC | 全部连续 | 284,806 |
| diamonds | 回归 | RMSE | 全部连续 | 53,940 |
| elevator | 回归 | RMSE | 全部连续 | 109,563 |
| housesale | 回归 | RMSE | 大部分分类 | 30,138 |

## 环境配置

### 1. 创建Conda环境

```bash
conda env create -f environment.yml
conda activate tabgnn
```

**注意**：当前环境使用 `tabgnn_clean`（Python 3.8），已配置好GPU支持。

### 2. 安装PyTorch CUDA版本（Linux GPU环境）

**当前配置**（CUDA 12.1）：
```bash
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121
pip install dgl -f https://data.dgl.ai/wheels/torch-2.1/cu121/repo.html
```

### 3. 验证安装

```python
import torch
import dgl
print(f"PyTorch版本: {torch.__version__}")
print(f"CUDA可用: {torch.cuda.is_available()}")
print(f"DGL版本: {dgl.__version__}")
```

## 数据格式

### CSV文件格式

- CSV文件：最后一列必须为标签列（列名为`TARGET`），其他列为特征
- 特征可以是数值型（NUMERIC）或分类型（CATEGORICAL）
- CSV文件第一行可以是列名（带header）或不带header
- 数据划分：默认 80/20 的训练/测试划分

### 数据集文件结构

每个数据集需要以下文件：

```
data/
├── test_data/
│   ├── your_dataset.csv              # 数据文件
│   └── your_dataset.ds_info.json     # 数据集信息（任务类型、特征类型等）
├── your_dataset/
│   └── your_dataset.db_info_fz.json  # 图数据库信息
└── tabular_ds_info.json              # 数据集注册文件
```

## 添加新数据集

详细的添加新数据集步骤请参考：[README_ADD_DATASET.md](README_ADD_DATASET.md)

**快速步骤**：
1. 准备CSV数据文件（最后一列是TARGET）
2. 创建 `ds_info.json`（定义任务类型和特征）
3. 创建 `db_info_fz.json`（定义图数据库结构）
4. 在 `tabular_ds_info.json` 中注册
5. 测试运行

### ds_info.json配置文件

每个数据集需要一个对应的`.ds_info.json`配置文件，放在`data/test_data/`目录下。

**示例：adult.ds_info.json**
```json
{
  "task": "binary classification",
  "columns": [
    {"name": "feature0", "type": "NUMERIC"},
    {"name": "feature1", "type": "NUMERIC"},
    {"name": "TARGET", "type": "CATEGORICAL", "cardinality": 2}
  ]
}
```

**字段说明：**
- `task`: 任务类型，支持 `"binary classification"`, `"multiclass classification"`, `"regression"`
- `columns`: 列定义数组
  - `name`: 列名（最后一列必须为`TARGET`）
  - `type`: 列类型，`"NUMERIC"` 或 `"CATEGORICAL"`
  - `cardinality`: 仅分类型列需要，表示类别数量

### 数据集注册

在`data/tabular_ds_info.json`中注册数据集：

```json
{
  "dataset_name": {
    "processed": {
      "task": "binary classification",
      "local_path": "test_data/dataset.csv",
      "ds_info": "data/test_data/dataset.ds_info.json"
    }
  }
}
```

## 使用方法

### 基本用法（推荐使用run.sh）

**修改 `run.sh` 中的超参数**（第18-26行），然后运行：

```bash
./run.sh
```

**或者通过命令行参数覆盖**：

```bash
./run.sh --dataset cardio --epochs 20 --batch_size 64 --device cuda
```

**直接使用python**：

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate tabgnn_clean
export LD_LIBRARY_PATH="/usr/local/cuda-12.2/targets/x86_64-linux/lib:/usr/local/cuda-12.2/lib64:$LD_LIBRARY_PATH"
python train.py --dataset adult --epochs 10 --batch_size 32 --device cuda
```

### 可用的数据集

- `adult` - 二分类任务
- `cardio` - 二分类任务
- `creditcard` - 二分类任务
- `diamonds` - 回归任务
- `elevator` - 回归任务
- `housesale` - 回归任务

### 训练参数

- `--dataset`: 数据集名称（必需）
- `--epochs`: 训练轮数（默认：10）
- `--batch_size`: 批次大小（默认：32）
- `--hidden_dim`: 隐藏层维度（默认：64）
- `--n_layers`: GNN层数（默认：3）
- `--dropout`: Dropout比例（默认：0.2）
- `--lr`: 学习率（默认：0.001）
- `--train_split`: 训练集比例（默认：0.8）
- `--device`: 设备（默认：cuda，或cpu）

### 完整参数说明

```bash
python train.py \
  --dataset adult \              # 数据集名称（必须在tabular_ds_info.json中注册）
  --epochs 10 \                  # 训练轮数（默认10）
  --batch_size 32 \              # 批次大小（默认32）
  --train_split 0.8 \            # 训练集比例（默认0.8）
  --hidden_dim 64 \              # 隐藏层维度（默认64）
  --n_layers 3 \                 # GNN层数（默认3）
  --fcout_layers 64 32 \         # 输出层MLP层大小（默认[64, 32]）
  --dropout 0.2 \                # Dropout比例（默认0.2）
  --lr 0.001 \                   # 学习率（默认0.001）
  --connect_keys col1 col2 \     # 连接键（分类特征名），None表示自动检测或使用相似性
  --max_neighbors 10 \           # 每个连接键的最大邻居数（默认10）
  --device cpu                   # 设备（默认cpu，可选cuda）
```

### 参数说明

**数据集参数：**
- `--dataset`: 数据集名称（必需）
- `--train_split`: 训练集比例（默认0.8）
- `--connect_keys`: 连接键列表（分类特征名），None表示自动检测或使用KNN相似性
- `--max_neighbors`: 每个连接键的最大邻居数（默认10）

**模型参数：**
- `--model`: 模型类型（默认GCN，目前仅支持GCN）
- `--hidden_dim`: 隐藏层维度（默认64）
- `--n_layers`: GNN层数（默认3）
- `--fcout_layers`: 输出层MLP层大小列表（默认[64, 32]）
- `--dropout`: Dropout比例（默认0.2）

**训练参数：**
- `--epochs`: 训练轮数（默认10）
- `--batch_size`: 批次大小（默认32）
- `--lr`: 学习率（默认0.001）
- `--device`: 设备，`cpu`或`cuda`（默认cpu）

## 运行示例

### 示例1：使用adult数据集（纯数值特征）

```bash
python train.py --dataset adult --epochs 10 --batch_size 32 --hidden_dim 64
```

这个示例会：
- 自动检测到没有分类特征，使用KNN相似性作为连接键
- 训练10个epoch
- 每个epoch输出训练时间、测试时间和评估指标（AUC）

### 示例2：指定连接键

```bash
python train.py --dataset your_dataset --connect_keys category1 category2 --epochs 20
```

这个示例会：
- 使用指定的分类特征作为连接键构建多重图
- 训练20个epoch

## 输出说明

训练过程中，每个epoch会输出：
- **训练信息**：训练损失、训练准确率（分类任务）、训练时间
- **测试信息**：测试损失、测试指标（AUC或RMSE）、测试时间
- **最佳标记**：`*`表示当前epoch的测试指标为最佳

**示例输出：**
```
Epoch 1/10: Train Loss: 0.4089, Train Acc: 0.8118 (246.70s) | Test Loss: 0.4165, Test AUC: 0.8868 (19.96s) *
Epoch 2/10: Train Loss: 0.3521, Train Acc: 0.8456 (245.12s) | Test Loss: 0.4012, Test AUC: 0.8923 (19.85s) *
...
```

## 评估指标

- **分类任务**：使用AUC（Area Under ROC Curve）作为评估指标
- **回归任务**：使用RMSE（Root Mean Squared Error）作为评估指标

## 项目结构

```
TabGNN/
├── train.py                  # 主训练脚本
├── environment.yml           # Conda环境配置
├── data/
│   ├── CSVToGraphAdapter.py # CSV到图数据适配器
│   ├── TabularDataset.py    # 表格数据集加载器
│   ├── tabular_ds_info.json # 数据集注册文件
│   └── test_data/           # 测试数据目录
│       ├── adult.csv
│       └── adult.ds_info.json
├── models/
│   └── GNN/
│       ├── GCN.py           # GCN模型实现
│       └── GNNModelBase.py  # GNN模型基类
└── utils.py                  # 工具函数（DGL collator等）
```

## 注意事项

1. **数据格式**：确保CSV文件最后一列为`TARGET`标签列
2. **内存使用**：大数据集构建KNN图可能占用较多内存，建议适当调整`--max_neighbors`
3. **CUDA版本**：在Linux GPU环境下需要替换PyTorch为CUDA版本
4. **连接键选择**：如果数据有分类特征，建议指定`--connect_keys`；纯数值数据会自动使用KNN相似性
5. **数据集注册**：新数据集需要在`data/tabular_ds_info.json`中注册

## 常见问题

**Q: 如何在Linux GPU环境下运行？**
A: 按照"环境配置"部分安装PyTorch CUDA版本，然后使用`--device cuda`参数。

**Q: 如何添加新数据集？**
A: 详细步骤请参考 [README_ADD_DATASET.md](README_ADD_DATASET.md)。快速步骤：1) 准备CSV文件和ds_info.json配置文件；2) 创建db_info_fz.json文件；3) 在`data/tabular_ds_info.json`中注册；4) 测试运行。

**Q: 连接键如何选择？**
A: 优先使用分类特征作为连接键；如果没有分类特征，系统会自动使用KNN相似性。

## 许可证

本项目基于原始TabGNN代码库修改。

