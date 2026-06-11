from __future__ import annotations

import os
import numpy as np
import pandas as pd


INPUT_FILE = "diamonds_mixed.csv"
OUTPUT_DIR = "diamonds_ood_experiments"
RANDOM_SEED = 42


def _pearson_corr(x: pd.Series, y: pd.Series) -> float:
    # pandas corr 默认 Pearson；若方差为 0 会返回 NaN（这里正常不会）
    return float(x.corr(y, method="pearson"))

def _to_markdown_table(df: pd.DataFrame) -> str:
    """
    Minimal markdown table renderer (avoid pandas.to_markdown dependency on tabulate).
    """
    headers = list(df.columns)
    rows = df.astype(str).values.tolist()

    widths = [len(h) for h in headers]
    for r in rows:
        for i, cell in enumerate(r):
            widths[i] = max(widths[i], len(cell))

    def fmt_row(vals: list[str]) -> str:
        return "| " + " | ".join(v.ljust(widths[i]) for i, v in enumerate(vals)) + " |"

    header_line = fmt_row(headers)
    align_line = "| " + " | ".join((":---").ljust(widths[i]) for i in range(len(headers))) + " |"
    body_lines = [fmt_row(r) for r in rows]
    return "\n".join([header_line, align_line] + body_lines)


def make_ood_split(
    df_all: pd.DataFrame,
    group_a_mask: pd.Series,
    *,
    seed: int,
    experiment_name: str,
    score_mean_spec: tuple[str, str] | None,
    corr_spec: tuple[str, str] | None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict, dict, int]:
    """
    用 N=min(|A|,|B|) 保证可以同时满足 Train 90/10 与 Test 10/90 且互斥：
      - Train: A_train=round(0.9N), B_train=N-A_train
      - Test : A_test=N-A_train,   B_test=A_train
    """
    rng = np.random.default_rng(seed)

    df_a = df_all.loc[group_a_mask].copy()
    df_b = df_all.loc[~group_a_mask].copy()
    n_a, n_b = len(df_a), len(df_b)
    if n_a == 0 or n_b == 0:
        raise ValueError(f"{experiment_name}: one group is empty (A={n_a}, B={n_b})")

    N = min(n_a, n_b)
    n_a_train = int(round(0.9 * N))
    n_a_test = N - n_a_train
    n_b_train = n_a_test
    n_b_test = n_a_train

    a_ids = rng.choice(df_a["row_id"].to_numpy(), size=N, replace=False)
    b_ids = rng.choice(df_b["row_id"].to_numpy(), size=N, replace=False)

    rng.shuffle(a_ids)
    rng.shuffle(b_ids)

    a_train_ids = set(a_ids[:n_a_train])
    a_test_ids = set(a_ids[n_a_train:])
    b_train_ids = set(b_ids[:n_b_train])
    b_test_ids = set(b_ids[n_b_train:])

    train_ids = a_train_ids | b_train_ids
    test_ids = a_test_ids | b_test_ids
    assert train_ids.isdisjoint(test_ids), f"{experiment_name}: train/test overlap detected"

    train_df = df_all[df_all["row_id"].isin(train_ids)].copy()
    test_df = df_all[df_all["row_id"].isin(test_ids)].copy()

    # 打乱行顺序（不改变互斥性）
    train_df = train_df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    test_df = test_df.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    def _summary(split_df: pd.DataFrame) -> dict:
        split_a_pct = 100.0 * float(split_df["row_id"].isin(a_ids).mean())
        split_b_pct = 100.0 - split_a_pct

        row = {
            "Experiment": experiment_name,
            "Split": "Train" if split_df is train_df else "Test",
            "Sample Count": len(split_df),
            "Group A %": f"{split_a_pct:.1f}%",
            "Group B %": f"{split_b_pct:.1f}%",
        }

        if score_mean_spec is not None:
            col, label = score_mean_spec
            row[label] = f"{float(split_df[col].mean()):.3f}"
        else:
            row["Target Mean"] = "-"

        if corr_spec is not None:
            xcol, ycol = corr_spec
            corr = _pearson_corr(split_df[xcol], split_df[ycol])
            row["Correlation (Clarity vs Price)"] = f"{corr:+.3f}"
        else:
            row["Correlation (Clarity vs Price)"] = "-"

        return row

    # 构造 summary（分别基于 train/test）
    train_row = _summary(train_df)
    test_row = _summary(test_df)
    return train_df, test_df, train_row, test_row, N


def main() -> None:
    print("=" * 80)
    print("Diamonds OOD/Causal Learning 实验数据生成")
    print("=" * 80)

    print("\n[1] 加载数据...")
    df_raw = pd.read_csv(INPUT_FILE)
    print(f"原始数据: {len(df_raw)} 行, {df_raw.shape[1]} 列")
    print(f"列名: {list(df_raw.columns)}")

    required_cols = ["carat", "cut", "color", "clarity", "depth", "table", "x", "y", "z", "price"]
    missing = [c for c in required_cols if c not in df_raw.columns]
    if missing:
        raise ValueError(f"Missing columns in {INPUT_FILE}: {missing}")

    print("\n[2] 去重 (Deduplication, 必须在切分前做)...")
    before = len(df_raw)
    df = df_raw.drop_duplicates().reset_index(drop=True)
    removed = before - len(df)
    print(f"去重前: {before} | 去重后: {len(df)} | 删除重复行: {removed}")

    # row_id 用于严格互斥检查（防止 reset_index 导致的“假互斥”）
    df["row_id"] = np.arange(len(df), dtype=np.int64)

    print("\n[3] 编码检查（来自 diamonds_mapping.csv 的序数编码约定）...")
    print(f"cut unique: {sorted(df['cut'].unique().tolist())}")
    print(f"color unique: {sorted(df['color'].unique().tolist())}")
    print(f"clarity unique: {sorted(df['clarity'].unique().tolist())}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    rows = []

    # -------------------------
    # Exp 1: Cut Shift
    # A: cut in {Premium(3), Ideal(4)}  <=> cut >= 3
    # -------------------------
    print("\n" + "=" * 80)
    print("Exp 1: Cut Shift")
    print("=" * 80)
    exp1_a = df["cut"] >= 3
    train1, test1, r1_train, r1_test, N1 = make_ood_split(
        df,
        exp1_a,
        seed=RANDOM_SEED + 1,
        experiment_name="Exp 1 (Cut Shift)",
        score_mean_spec=("cut", "Cut Mean"),
        corr_spec=None,
    )
    train1.drop(columns=["row_id"]).to_csv(os.path.join(OUTPUT_DIR, "train_exp1.csv"), index=False)
    test1.drop(columns=["row_id"]).to_csv(os.path.join(OUTPUT_DIR, "test_exp1.csv"), index=False)
    rows += [r1_train, r1_test]
    print(f"采样规模 N=min(|A|,|B|)={N1} → train/test 各 {N1} 行")

    # -------------------------
    # Exp 2: Color Shift
    # A: color in {D(6),E(5),F(4)} <=> color >= 4
    # -------------------------
    print("\n" + "=" * 80)
    print("Exp 2: Color Shift")
    print("=" * 80)
    exp2_a = df["color"] >= 4
    train2, test2, r2_train, r2_test, N2 = make_ood_split(
        df,
        exp2_a,
        seed=RANDOM_SEED + 2,
        experiment_name="Exp 2 (Color Shift)",
        score_mean_spec=("color", "Color Mean"),
        corr_spec=None,
    )
    train2.drop(columns=["row_id"]).to_csv(os.path.join(OUTPUT_DIR, "train_exp2.csv"), index=False)
    test2.drop(columns=["row_id"]).to_csv(os.path.join(OUTPUT_DIR, "test_exp2.csv"), index=False)
    rows += [r2_train, r2_test]
    print(f"采样规模 N=min(|A|,|B|)={N2} → train/test 各 {N2} 行")

    # -------------------------
    # Exp 3: Simpson's Paradox
    # median split: carat (size) and clarity (clarity_score)
    # A (spurious/negative): (small & high) OR (big & low)
    # B (causal/positive):  (small & low)  OR (big & high)
    # -------------------------
    print("\n" + "=" * 80)
    print("Exp 3: Simpson's Paradox")
    print("=" * 80)
    median_carat = float(df["carat"].median())
    median_clarity = float(df["clarity"].median())
    print(f"median(carat)={median_carat:.4f} | median(clarity_score)={median_clarity:.4f}")

    small = df["carat"] < median_carat
    high = df["clarity"] >= median_clarity
    exp3_a = (small & high) | ((~small) & (~high))  # A = spurious

    train3, test3, r3_train, r3_test, N3 = make_ood_split(
        df,
        exp3_a,
        seed=RANDOM_SEED + 3,
        experiment_name="Exp 3 (Simpson)",
        score_mean_spec=None,
        corr_spec=("clarity", "price"),
    )
    train3.drop(columns=["row_id"]).to_csv(os.path.join(OUTPUT_DIR, "train_exp3.csv"), index=False)
    test3.drop(columns=["row_id"]).to_csv(os.path.join(OUTPUT_DIR, "test_exp3.csv"), index=False)
    rows += [r3_train, r3_test]
    print(f"采样规模 N=min(|A|,|B|)={N3} → train/test 各 {N3} 行")

    # -------------------------
    # Verification table
    # -------------------------
    print("\n" + "=" * 80)
    print("验证表格 (Markdown)")
    print("=" * 80)

    df_summary = pd.DataFrame(rows)
    # 让列顺序更贴近需求
    col_order = [
        "Experiment",
        "Split",
        "Sample Count",
        "Group A %",
        "Group B %",
        "Cut Mean",
        "Color Mean",
        "Correlation (Clarity vs Price)",
    ]
    for c in col_order:
        if c not in df_summary.columns:
            df_summary[c] = "-"
    df_summary = df_summary[col_order]

    md_table = _to_markdown_table(df_summary)
    print("\n" + md_table)

    df_summary.to_csv(os.path.join(OUTPUT_DIR, "verification_table.csv"), index=False)
    with open(os.path.join(OUTPUT_DIR, "verification_table.md"), "w", encoding="utf-8") as f:
        f.write(md_table + "\n")

    # -------------------------
    # Design doc
    # -------------------------
    design_md = f"""# Diamonds OOD / Causal Learning（Soft Shifts / Selection Bias）实验设计

## 数据来源与编码

- 输入文件：`{INPUT_FILE}`（已按 `diamonds_mapping.csv` 做序数编码）
- cut: Fair(0) < Good(1) < Very Good(2) < Premium(3) < Ideal(4)
- color: J(0) < I(1) < H(2) < G(3) < F(4) < E(5) < D(6)
- clarity: I1(0) < SI2(1) < SI1(2) < VS2(3) < VS1(4) < VVS2(5) < VVS1(6) < IF(7)

## 去重（防止数据泄漏）

切分前去重：去重前 {before} 行 → 去重后 {len(df)} 行（删除 {removed} 行重复样本）。

## 全局采样策略（9:1 ↔ 1:9）

对每个实验定义 A/B 两组，然后用 **N = min(|A|, |B|)** 构造互斥切分：

- Train: 90% A + 10% B
- Test : 10% A + 90% B
- 严格互斥：通过 `row_id` 检查 train/test 无交集

这样 train/test 都有 N 条样本，并且 A/B 的比例按设计“强反转”。

## Exp 1：Cut Shift（单特征偏移）

- A（高切工）：cut ∈ {{Premium(3), Ideal(4)}}（即 cut ≥ 3）
- B（低切工）：cut ∈ {{Fair, Good, Very Good}}（即 cut ≤ 2）

## Exp 2：Color Shift（单特征偏移）

- A（高色级）：color ∈ {{D(6), E(5), F(4)}}（即 color ≥ 4）
- B（低色级）：color ∈ {{G, H, I, J}}（即 color ≤ 3）

## Exp 3：Simpson’s Paradox（虚假相关性反转）

使用中位数切分：

- median(carat) = {median_carat:.4f}
- median(clarity_score) = {median_clarity:.4f}

定义：

- A（虚假相关/简单模式）：(小克拉 且 高净度) 或 (大克拉 且 低净度)
- B（因果机制/困难模式）：(小克拉 且 低净度) 或 (大克拉 且 高净度)

验证指标：Pearson corr(clarity_score, price) 在 Train vs Test 应出现显著差异（理想为负→正的翻转）。

## 输出文件

输出目录：`{OUTPUT_DIR}/`

- `train_exp1.csv`, `test_exp1.csv`
- `train_exp2.csv`, `test_exp2.csv`
- `train_exp3.csv`, `test_exp3.csv`
- `verification_table.csv`, `verification_table.md`

验证表格：

{md_table}
"""

    with open(os.path.join(OUTPUT_DIR, "EXPERIMENT_DESIGN.md"), "w", encoding="utf-8") as f:
        f.write(design_md)

    print(f"\n✓ 输出目录: {OUTPUT_DIR}/ （已生成 6 个 CSV + verification_table + EXPERIMENT_DESIGN.md）")


if __name__ == "__main__":
    main()
