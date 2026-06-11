# 如何添加新数据集到TabGNN

本文档详细说明如何将新数据集添加到TabGNN训练框架中。

## 步骤概述

添加新数据集需要完成以下步骤：
1. 准备数据文件（CSV格式）
2. 创建 `ds_info.json` 文件（定义任务类型和特征信息）
3. 创建 `db_info_fz.json` 文件（定义图数据库结构）
4. 在 `tabular_ds_info.json` 中注册数据集
5. 测试数据集是否能正常运行

---

## 详细步骤

### 1. 准备数据文件

#### 1.1 数据格式要求

- **CSV格式**：最后一列必须是标签列，列名为 `TARGET`
- **特征列**：除最后一列外的所有列都是特征
- **列名**：
  - 如果有列名（header），第一行是列名
  - 如果没有列名，使用默认的 `feature0, feature1, ..., featureN-1, TARGET`

#### 1.2 复制数据文件

将数据文件复制到 `data/test_data/` 或 `data/raw_data/` 目录：

```bash
cp /path/to/your/dataset.csv data/test_data/your_dataset.csv
```

### 2. 创建 `ds_info.json` 文件

在 `data/test_data/` 目录下创建 `your_dataset.ds_info.json` 文件。

#### 2.1 二分类任务示例

```json
{
  "task": "binary classification",
  "columns": [
    {
      "name": "feature0",
      "type": "NUMERIC"  // 或 "CATEGORICAL"
    },
    {
      "name": "feature1",
      "type": "CATEGORICAL",
      "cardinality": 5  // 分类特征需要指定类别数
    },
    ...
    {
      "name": "TARGET",
      "type": "CATEGORICAL",
      "cardinality": 2
    }
  ]
}
```

#### 2.2 回归任务示例

```json
{
  "task": "regression",
  "columns": [
    {
      "name": "feature0",
      "type": "NUMERIC"
    },
    ...
    {
      "name": "TARGET",
      "type": "NUMERIC"
    }
  ]
}
```

#### 2.3 特征类型说明

- **NUMERIC**: 连续数值特征（整数或浮点数）
- **CATEGORICAL**: 分类特征
  - 必须指定 `cardinality`（类别数）
  - 例如：`"cardinality": 5` 表示该特征有5个不同的值

#### 2.4 快速生成 `ds_info.json`

可以使用以下Python脚本自动生成（需要根据实际情况调整）：

```python
import pandas as pd
import json

# 读取数据集
df = pd.read_csv('data/test_data/your_dataset.csv', header=None)  # 或 header=0 如果有列名
num_cols = len(df.columns)

# 确定任务类型（根据TARGET列的特征判断）
# 如果TARGET是连续值 -> regression
# 如果TARGET是分类值 -> binary classification 或 multiclass classification

task_type = "regression"  # 或 "binary classification"

columns = []
for i in range(num_cols - 1):
    # 判断特征类型
    unique_count = df.iloc[:, i].nunique()
    if unique_count < 100:  # 如果唯一值少，可能是分类特征
        columns.append({
            "name": f"feature{i}",
            "type": "CATEGORICAL",
            "cardinality": int(unique_count)
        })
    else:
        columns.append({
            "name": f"feature{i}",
            "type": "NUMERIC"
        })

# 添加TARGET列
if task_type == "regression":
    columns.append({
        "name": "TARGET",
        "type": "NUMERIC"
    })
else:
    target_unique = df.iloc[:, -1].nunique()
    columns.append({
        "name": "TARGET",
        "type": "CATEGORICAL",
        "cardinality": int(target_unique)
    })

ds_info = {
    "task": task_type,
    "columns": columns
}

# 保存
with open('data/test_data/your_dataset.ds_info.json', 'w') as f:
    json.dump(ds_info, f, indent=2)
```

### 3. 创建 `db_info_fz.json` 文件

在 `data/your_dataset/` 目录下创建 `your_dataset.db_info_fz.json` 文件。

#### 3.1 二分类任务示例

```json
{
 "task": {
  "type": "classification",
  "n_classes": 2,
  "n_train": 8000,
  "n_test": 2000,
  "train_class_counts": [6000, 2000]
 },
 "node_type_to_int": {
  "Main_table": 0
 },
 "edge_type_to_int": {
  "SELF": 0,
  "SIMILARITY_EDGE": 1
 },
 "node_types_and_features": {
  "Main_table": {
   "INDEX_ID": {
    "type": "SCALAR"
   },
   "feature0": {
    "type": "SCALAR"  // 连续特征用 "SCALAR"
   },
   "feature1": {
    "type": "CATEGORICAL",  // 分类特征用 "CATEGORICAL"
    "cardinality": 5
   },
   ...
   "TARGET": {
    "type": "CATEGORICAL",
    "cardinality": 2
   }
  }
 },
 "label_feature": "Main_table.TARGET"
}
```

#### 3.2 回归任务示例

```json
{
 "task": {
  "type": "regression",
  "n_classes": 1,
  "n_train": 8000,
  "n_test": 2000,
  "train_class_counts": []
 },
 "node_type_to_int": {
  "Main_table": 0
 },
 "edge_type_to_int": {
  "SELF": 0,
  "SIMILARITY_EDGE": 1
 },
 "node_types_and_features": {
  "Main_table": {
   "INDEX_ID": {
    "type": "SCALAR"
   },
   "feature0": {
    "type": "SCALAR"
   },
   ...
   "TARGET": {
    "type": "SCALAR"
   }
  }
 },
 "label_feature": "Main_table.TARGET"
}
```

#### 3.3 自动生成 `db_info_fz.json`

可以使用以下Python脚本自动生成：

```python
import pandas as pd
import json
import numpy as np

# 读取数据集
df = pd.read_csv('data/test_data/your_dataset.csv', header=None)  # 根据实际情况调整
n_total = len(df)
train_split = 0.8
n_train = int(n_total * train_split)
n_test = n_total - n_train

# 读取ds_info
with open('data/test_data/your_dataset.ds_info.json', 'r') as f:
    ds_info = json.load(f)

# 统计训练集类别分布（仅分类任务需要）
if ds_info['task'] == 'regression':
    train_class_counts = []
    task_type_db = "regression"
    n_classes = 1
else:
    train_targets = df.iloc[:n_train, -1]
    unique_targets = sorted(train_targets.unique())
    train_class_counts = [int((train_targets == t).sum()) for t in unique_targets]
    task_type_db = "classification"
    n_classes = len(unique_targets)

# 创建db_info
db_info = {
    "task": {
        "type": task_type_db,
        "n_classes": n_classes,
        "n_train": n_train,
        "n_test": n_test,
        "train_class_counts": train_class_counts
    },
    "node_type_to_int": {
        "Main_table": 0
    },
    "edge_type_to_int": {
        "SELF": 0,
        "SIMILARITY_EDGE": 1
    },
    "node_types_and_features": {
        "Main_table": {
            "INDEX_ID": {
                "type": "SCALAR"
            }
        }
    },
    "label_feature": "Main_table.TARGET"
}

# 添加特征
for col in ds_info['columns']:
    if col['name'] != 'TARGET':
        if col['type'] == 'NUMERIC':
            db_info["node_types_and_features"]["Main_table"][col['name']] = {
                "type": "SCALAR"
            }
        else:  # CATEGORICAL
            db_info["node_types_and_features"]["Main_table"][col['name']] = {
                "type": "CATEGORICAL",
                "cardinality": col['cardinality']
            }

# 添加TARGET
target_col = ds_info['columns'][-1]
if target_col['type'] == 'NUMERIC':
    db_info["node_types_and_features"]["Main_table"]["TARGET"] = {
        "type": "SCALAR"
    }
else:
    db_info["node_types_and_features"]["Main_table"]["TARGET"] = {
        "type": "CATEGORICAL",
        "cardinality": target_col['cardinality']
    }

# 创建目录并保存
import os
os.makedirs('data/your_dataset', exist_ok=True)
with open('data/your_dataset/your_dataset.db_info_fz.json', 'w') as f:
    json.dump(db_info, f, indent=1)
```

### 4. 在 `tabular_ds_info.json` 中注册

编辑 `data/tabular_ds_info.json`，添加新数据集的条目：

```json
{
  ...
  "your_dataset": {
    "processed": {
      "task": "binary classification",  // 或 "regression"
      "local_path": "test_data/your_dataset.csv",
      "ds_info": "data/test_data/your_dataset.ds_info.json"
    }
  }
}
```

### 5. 处理CSV文件格式

如果CSV文件有列名（header），需要在 `TabularDataset.py` 中添加特殊处理：

```python
# 在 TabularDataset.py 的 __init__ 方法中
elif dataset_name in ['elevator']:
    # 第一行是列名，需要跳过
    self.raw_data = pd.read_csv(raw_data_path, header=0, names=col_names)
elif dataset_name in ['housesale', 'creditcard']:
    # 有header，使用header=0并重命名列
    self.raw_data = pd.read_csv(raw_data_path, header=0)
    if len(self.raw_data.columns) == len(col_names):
        self.raw_data.columns = col_names
```

### 6. 测试数据集

#### 6.1 创建小数据集用于快速测试

```bash
head -1000 data/test_data/your_dataset.csv > data/test_data/your_dataset_test_small.csv
```

#### 6.2 测试运行

修改 `tabular_ds_info.json` 临时使用小数据集，然后运行：

```bash
./run.sh --dataset your_dataset --epochs 1 --batch_size 32 --device cuda
```

如果测试通过，恢复为完整数据集路径。

---

## 完整示例：添加新数据集 `example`

### 步骤1：准备数据

```bash
cp /path/to/example.csv data/test_data/example.csv
```

### 步骤2：创建 `example.ds_info.json`

```json
{
  "task": "binary classification",
  "columns": [
    {"name": "feature0", "type": "NUMERIC"},
    {"name": "feature1", "type": "CATEGORICAL", "cardinality": 3},
    {"name": "TARGET", "type": "CATEGORICAL", "cardinality": 2}
  ]
}
```

### 步骤3：创建 `example.db_info_fz.json`

```json
{
 "task": {
  "type": "classification",
  "n_classes": 2,
  "n_train": 8000,
  "n_test": 2000,
  "train_class_counts": [5000, 3000]
 },
 "node_type_to_int": {"Main_table": 0},
 "edge_type_to_int": {"SELF": 0, "SIMILARITY_EDGE": 1},
 "node_types_and_features": {
  "Main_table": {
   "INDEX_ID": {"type": "SCALAR"},
   "feature0": {"type": "SCALAR"},
   "feature1": {"type": "CATEGORICAL", "cardinality": 3},
   "TARGET": {"type": "CATEGORICAL", "cardinality": 2}
  }
 },
 "label_feature": "Main_table.TARGET"
}
```

### 步骤4：注册到 `tabular_ds_info.json`

```json
{
  ...
  "example": {
    "processed": {
      "task": "binary classification",
      "local_path": "test_data/example.csv",
      "ds_info": "data/test_data/example.ds_info.json"
    }
  }
}
```

### 步骤5：测试

```bash
./run.sh --dataset example --epochs 1 --batch_size 32 --device cuda
```

---

## 注意事项

1. **数据划分**：默认使用 80/20 的训练/测试划分，与 Causal_attention 保持一致
2. **连接键**：如果没有分类特征，会自动使用相似性连接（`__similarity__`）
3. **任务类型**：
   - 二分类：使用 AUC 作为评估指标
   - 回归：使用 RMSE 作为评估指标
4. **特征编码**：
   - NUMERIC 特征使用 `ScalarRobustScalerEnc` 编码
   - CATEGORICAL 特征使用 `ScalarQuantileOrdinalEnc` 编码

---

## 参考：已适配的数据集

当前已适配的数据集及其配置：

1. **cardio** - 二分类，所有特征为分类特征
2. **creditcard** - 二分类，所有特征为连续特征
3. **diamonds** - 回归，所有特征为连续特征
4. **elevator** - 回归，所有特征为连续特征（使用elevator_cleaned.csv）
5. **housesale** - 回归，大部分特征为分类特征
6. **adult** - 二分类，所有特征为连续特征

可以参考这些数据集的配置文件来创建新数据集的配置。

