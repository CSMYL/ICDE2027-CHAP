# Diamonds OOD / Causal Learning（Soft Shifts / Selection Bias）实验设计

## 数据来源与编码

- 输入文件：`diamonds_mixed.csv`（已按 `diamonds_mapping.csv` 做序数编码）
- cut: Fair(0) < Good(1) < Very Good(2) < Premium(3) < Ideal(4)
- color: J(0) < I(1) < H(2) < G(3) < F(4) < E(5) < D(6)
- clarity: I1(0) < SI2(1) < SI1(2) < VS2(3) < VS1(4) < VVS2(5) < VVS1(6) < IF(7)

## 去重（防止数据泄漏）

切分前去重：去重前 53940 行 → 去重后 53794 行（删除 146 行重复样本）。

## 全局采样策略（9:1 ↔ 1:9）

对每个实验定义 A/B 两组，然后用 **N = min(|A|, |B|)** 构造互斥切分：

- Train: 90% A + 10% B
- Test : 10% A + 90% B
- 严格互斥：通过 `row_id` 检查 train/test 无交集

这样 train/test 都有 N 条样本，并且 A/B 的比例按设计“强反转”。

## Exp 1：Cut Shift（单特征偏移）

- A（高切工）：cut ∈ {Premium(3), Ideal(4)}（即 cut ≥ 3）
- B（低切工）：cut ∈ {Fair, Good, Very Good}（即 cut ≤ 2）

## Exp 2：Color Shift（单特征偏移）

- A（高色级）：color ∈ {D(6), E(5), F(4)}（即 color ≥ 4）
- B（低色级）：color ∈ {G, H, I, J}（即 color ≤ 3）

## Exp 3：Simpson’s Paradox（虚假相关性反转）

使用中位数切分：

- median(carat) = 0.7000
- median(clarity_score) = 3.0000

定义：

- A（虚假相关/简单模式）：(小克拉 且 高净度) 或 (大克拉 且 低净度)
- B（因果机制/困难模式）：(小克拉 且 低净度) 或 (大克拉 且 高净度)

验证指标：Pearson corr(clarity_score, price) 在 Train vs Test 应出现显著差异（理想为负→正的翻转）。

## 输出文件

输出目录：`diamonds_ood_experiments/`

- `train_exp1.csv`, `test_exp1.csv`
- `train_exp2.csv`, `test_exp2.csv`
- `train_exp3.csv`, `test_exp3.csv`
- `verification_table.csv`, `verification_table.md`

验证表格：

| Experiment          | Split | Sample Count | Group A % | Group B % | Cut Mean | Color Mean | Correlation (Clarity vs Price) |
| :---                | :---  | :---         | :---      | :---      | :---     | :---       | :---                           |
| Exp 1 (Cut Shift)   | Train | 18558        | 90.0%     | 10.0%     | 3.406    | nan        | -                              |
| Exp 1 (Cut Shift)   | Test  | 18558        | 10.0%     | 90.0%     | 1.770    | nan        | -                              |
| Exp 2 (Color Shift) | Train | 26051        | 90.0%     | 10.0%     | nan      | 4.604      | -                              |
| Exp 2 (Color Shift) | Test  | 26051        | 10.0%     | 90.0%     | nan      | 2.299      | -                              |
| Exp 3 (Simpson)     | Train | 19355        | 90.0%     | 10.0%     | nan      | nan        | -0.416                         |
| Exp 3 (Simpson)     | Test  | 19355        | 10.0%     | 90.0%     | nan      | nan        | +0.443                         |
