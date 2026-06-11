# LLM-CD Baseline 交接说明

这个目录实现了论文 baseline **LLM-CD**：

> Causal Discovery through Synergizing Large Language Model and Data-Driven Reasoning

目标是尽量还原原论文“数据驱动因果发现 + 大模型因果判断 + 只用目标父节点训练预测器”的流程，同时适配本项目 CHAP 数据格式和无网络服务器环境。

核心流程：

1. 用 PC 算法从表格数据发现初始因果图。
2. 用 LLM 判断不确定的条件独立关系、边方向和环处理。
3. 得到最终图后，只取目标变量的直接父节点。
4. 用这些父节点训练 MLP，并在测试集评估。

由于正式服务器可能无网络，流程被拆成三个阶段：第一阶段和第三阶段可以在无网服务器跑，只有第二阶段调用 LLM API 时需要有网。

## 5 分钟快速跑通

先在项目根目录确认环境。Mac 本地已经用 `.venv` 跑通过；新机器可以按下面装：

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements-baseline.txt
```

跑一个不调用 API 的 demo：

```bash
baseline/llmcd/scripts/run_e2e.sh baseline/llmcd/configs/synthetic_demo.json
```

输出在：

```text
baseline/llmcd/runs/synthetic_demo/
```

应该看到三个文件：

- `pc_graph.json`：PC 初始图
- `final_graph.json`：LLM 或 fallback 规则修正后的最终图
- `metrics.json`：只用目标父节点训练 MLP 后的测试指标

## 正式数据集配置

已经准备好 6 个正式配置：

| 数据集 | 配置文件 | target |
| --- | --- | --- |
| Adult | `baseline/llmcd/configs/adult.json` | `income`，第 14 列 |
| Cardio | `baseline/llmcd/configs/cardio.json` | `cardio`，第 11 列 |
| CreditCard | `baseline/llmcd/configs/creditcard.json` | `Class`，第 30 列 |
| Diamonds | `baseline/llmcd/configs/diamonds.json` | `price`，第 9 列 |
| Elevator | `baseline/llmcd/configs/elevator.json` | `vibration`，第 7 列 |
| Housesale | `baseline/llmcd/configs/housesale.json` | `Price`，第 39 列 |

这些配置都走项目已有 loader，不直接重新读原始 CSV。这样能保持和 CHAP 主实验一致的预处理、列顺序和 target 位置。

已验证的本地 shape：

```text
adult       (48842, 15)   target 14 income
cardio      (70000, 12)   target 11 cardio
creditcard  (284806, 31)  target 30 Class
diamonds    (53940, 10)   target 9  price
elevator    (109563, 8)   target 7  vibration
housesale   (30138, 40)   target 39 Price
```

正式配置默认最后训练 MLP：

```json
"training": {
  "predictor": "mlp"
}
```

## 一条命令端到端

不调用 LLM，只用 fallback 规则跑完整流程：

```bash
baseline/llmcd/scripts/run_e2e.sh baseline/llmcd/configs/adult.json
```

调用 LLM 跑完整流程：

```bash
export LLMCD_API_KEY="你的 API key"
export LLMCD_BASE_URL="https://api.deepseek.com"
export LLMCD_MODEL="deepseek-v4-flash"
export USE_LLM=1

baseline/llmcd/scripts/run_e2e.sh baseline/llmcd/configs/adult.json
```

注意：不要把真实 API key 写进 json、py、sh 文件。只通过环境变量传入。

## 无网服务器推荐流程

如果服务器无网，推荐分三步跑。

先打印分步命令：

```bash
baseline/llmcd/scripts/step_by_step_commands.sh baseline/llmcd/configs/adult.json
```

### 第一步：无网服务器跑 PC

```bash
.venv/bin/python baseline/llmcd/discover_pc_graph.py \
  --config baseline/llmcd/configs/adult.json \
  --run_dir baseline/llmcd/runs/adult_llmcd
```

输出：

```text
baseline/llmcd/runs/adult_llmcd/pc_graph.json
```

把下面两个文件复制到有网机器：

- `baseline/llmcd/configs/adult.json`
- `baseline/llmcd/runs/adult_llmcd/pc_graph.json`

### 第二步：有网机器调用 LLM

```bash
export LLMCD_API_KEY="你的 API key"
export LLMCD_BASE_URL="https://api.deepseek.com"
export LLMCD_MODEL="deepseek-v4-flash"

.venv/bin/python baseline/llmcd/judge_llm_graph.py \
  --config baseline/llmcd/configs/adult.json \
  --pc_graph baseline/llmcd/runs/adult_llmcd/pc_graph.json \
  --run_dir baseline/llmcd/runs/adult_llmcd \
  --use_llm
```

输出：

```text
baseline/llmcd/runs/adult_llmcd/final_graph.json
```

把 `final_graph.json` 复制回无网服务器。

### 第三步：无网服务器训练和测试

```bash
.venv/bin/python baseline/llmcd/train_eval_from_graph.py \
  --config baseline/llmcd/configs/adult.json \
  --graph baseline/llmcd/runs/adult_llmcd/final_graph.json \
  --run_dir baseline/llmcd/runs/adult_llmcd
```

输出：

```text
baseline/llmcd/runs/adult_llmcd/metrics.json
```

## 文件结构

```text
baseline/llmcd/
  configs/
    adult.json
    cardio.json
    creditcard.json
    diamonds.json
    elevator.json
    housesale.json
    synthetic_demo.json
    template.json
  scripts/
    run_e2e.sh
    step_by_step_commands.sh
  runs/
    .gitkeep
    <run_name>/
      pc_graph.json
      final_graph.json
      metrics.json
  common.py
  discover_pc_graph.py
  judge_llm_graph.py
  train_eval_from_graph.py
  .env.example
```

项目根目录还需要这些文件：

- `baseline/llm_cd_baseline.py`：复用已有数据 loader、PC 辅助函数、父节点 predictor 评估函数。
- `baseline/llm_cd_prompts.py`：LLM-CD prompt 和 API 调用逻辑。
- `requirements-baseline.txt`：baseline 依赖。

## 每个阶段的输入输出

### 阶段 1：数据到 PC 图

脚本：

```bash
.venv/bin/python baseline/llmcd/discover_pc_graph.py \
  --config <config.json> \
  --run_dir <run_dir>
```

输出 `pc_graph.json`，主要字段：

- `pc_directed_edges`：PC 发现的有向边
- `pc_undirected_edges`：PC 保留的无向边
- `uncertain_ci_pairs`：接近显著性阈值的候选 CI pair
- `feature_names`：变量名
- `feature_descriptions`：给 LLM 的变量说明

### 阶段 2：PC 图到最终图

调用 LLM：

```bash
.venv/bin/python baseline/llmcd/judge_llm_graph.py \
  --config <config.json> \
  --pc_graph <run_dir>/pc_graph.json \
  --run_dir <run_dir> \
  --use_llm
```

无网 fallback：

```bash
.venv/bin/python baseline/llmcd/judge_llm_graph.py \
  --config <config.json> \
  --pc_graph <run_dir>/pc_graph.json \
  --run_dir <run_dir>
```

输出 `final_graph.json`，主要字段：

- `graph`：最终邻接矩阵
- `parents`：目标节点直接父节点下标
- `parent_names`：目标节点直接父节点名字
- `ci_decisions`：LLM 对 CI pair 的判断
- `edge_decisions`：LLM 对边的方向、保留、删除判断

### 阶段 3：最终图到训练测试

```bash
.venv/bin/python baseline/llmcd/train_eval_from_graph.py \
  --config <config.json> \
  --graph <run_dir>/final_graph.json \
  --run_dir <run_dir>
```

输出 `metrics.json`，其中 `metrics.test` 是最终测试结果。

## 配置怎么改

复制模板：

```bash
cp baseline/llmcd/configs/template.json baseline/llmcd/configs/my_dataset.json
```

如果使用项目已有 CHAP loader，只需要改：

```json
{
  "run_name": "adult_llmcd",
  "dataset": "adult"
}
```

可用 dataset 名称：

- `adult`
- `cardio`
- `creditcard`
- `diamonds`
- `diamonds_mixed`
- `elevator`
- `housesale`
- `housing`
- `synthetic`

如果使用自定义 CSV，填写 `data`：

```json
{
  "run_name": "my_table_llmcd",
  "dataset": "my_table",
  "data": {
    "table_path": "raw_data/my_table.csv",
    "header": "infer",
    "target_column": "label",
    "target_idx": null,
    "categorical_indices": [1, 3],
    "continuous_indices": [0, 2, 4],
    "standardize_continuous": true,
    "drop_columns": []
  }
}
```

字段说明：

- `table_path`：CSV 路径。如果为空，走项目已有 loader。
- `header`：有表头写 `"infer"`；没有表头写 `null`。
- `target_column`：目标列名。有表头时推荐用这个。
- `target_idx`：目标列下标。从 0 开始。
- `categorical_indices`：分类变量列下标。
- `continuous_indices`：连续变量列下标。
- `standardize_continuous`：是否标准化连续变量。
- `drop_columns`：不参与实验的列名。

LLM-CD 依赖变量语义，建议认真写 `features`：

```json
{
  "features": [
    {
      "index": 0,
      "name": "age",
      "description": "The person's age."
    }
  ]
}
```

`index` 必须和数据矩阵列下标一致。

## 数据集语义说明

几个需要接手人注意的点：

- `adult`：本项目实际读入的是处理后的数值矩阵，配置里的说明写的是 Adult Census 原始字段语义。
- `cardio`：CHAP 预处理把若干连续医学变量离散化。配置里的说明写原始医学含义，但 PC 和训练使用处理后的数值。
- `creditcard`：`V1` 到 `V28` 是匿名 PCA 特征，无法知道真实业务含义。LLM 在这类数据上的优势会天然受限。
- `diamonds`：`price` 被移动到最后一列作为回归目标。
- `elevator`：`vibration` 被移动到最后一列作为回归目标；`x1` 到 `x5` 是匿名或工程传感器特征。
- `housesale`：6 个城市 CSV 合并后按 seed 42 shuffle，`Price` 是最后一列回归目标。

## PC 参数

配置中：

```json
"pc": {
  "alpha": 0.05,
  "independence_test": "auto",
  "sample_size": 2000,
  "ci_threshold": 0.001,
  "max_uncertain_pairs": 100,
  "jitter_scale": 0.000001
}
```

常用改法：

- PC 太慢：降低 `sample_size`。
- 高维数据太慢：先把 `sample_size` 调到 `500` 做管线验证。
- 奇异矩阵报错：代码会自动用极小 `jitter_scale` 重跑 PC。
- 想减少 LLM 调用量：降低 `max_uncertain_pairs`。

当前 `creditcard` 和 `housesale` 的 `sample_size` 已设为 `500`，因为它们维度较高，本地 Mac 上用 `2000` 跑 PC 会明显变慢。训练阶段仍使用完整数据。

## Predictor 参数

配置中：

```json
"training": {
  "predictor": "mlp"
}
```

可选：

- `mlp`：MLPClassifier 或 MLPRegressor，正式配置默认值。
- `rf`：RandomForest，分类和回归都能用。
- `logistic`：分类任务 LogisticRegression。
- `linear`：回归任务 LinearRegression。

为了实验一致性，正式 baseline 建议所有数据集保持同一个 predictor 设置。

## 已在 Mac 本地验证

本地已经跑通过：

- 6 个正式配置的 JSON 语法检查
- 6 个正式配置和 loader 的 shape、target 下标、任务类型检查
- 6 个数据集的 PC 图生成
- 6 个数据集的 fallback final graph 生成
- 6 个数据集的 MLP 训练和 metrics 输出

fallback 跑出的父节点示例：

```text
adult       -> marital_status, sex, capital_gain, capital_loss
cardio      -> age_group, systolic_pressure_bin, cholesterol
creditcard  -> V11, V14, V25
diamonds    -> color, clarity
elevator    -> x4
housesale   -> Location, No_of_Bedrooms, Resale
```

这些结果只是无 API fallback 的冒烟验证，不等价于正式 LLM-CD 结果。正式实验应在第二阶段加 `--use_llm`。

## 常见问题

### API 调不通

检查：

```bash
echo "$LLMCD_BASE_URL"
echo "$LLMCD_MODEL"
test -n "$LLMCD_API_KEY" && echo "api key is set"
```

不要把 key 打印出来，也不要写进文件。

### 服务器没有网络

只在无网服务器跑阶段 1 和阶段 3。阶段 2 拿到有网机器跑，产物 `final_graph.json` 再复制回来。

### PC 很慢

优先调小配置里的 `pc.sample_size`。这只影响因果图发现阶段，不影响最终 MLP 训练使用完整数据。

### 没有列名或列名匿名

可以跑，但 LLM 判断会弱。`creditcard` 的 PCA 特征就是这种情况。

## 和原论文的差异

这个实现没有修改 `causal-learn` 内部源码，而是把 LLM 介入放在 PC 图之后的独立阶段。这样更容易在无网服务器和有网机器之间搬运中间文件，也更方便接手人调试；代价是 skeleton 阶段的 LLM 介入不是完全嵌入 PC 搜索过程。
