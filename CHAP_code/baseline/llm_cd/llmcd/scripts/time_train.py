"""计时脚本：测 creditcard / elevator / meps 前5 epoch 训练和推理时间"""
import time, json, numpy as np
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.model_selection import train_test_split
import sys
sys.path.insert(0, ".")
from baseline.llm_cd_baseline import load_baseline_arrays

DATASETS = ["creditcard", "elevator", "meps"]

for ds in DATASETS:
    print(f"\n===== {ds} =====")
    full, y, v, num_classes_dict, target_idx = load_baseline_arrays(ds)

    g = json.load(open(f"baseline/llmcd/runs/{ds}_llmcd/final_graph.json"))
    parents = g["parents"]
    x = full[:, parents]
    is_clf = v[target_idx] == 1

    x_tr, x_te, y_tr, y_te = train_test_split(x, y, test_size=0.2, random_state=42)

    if is_clf:
        model = MLPClassifier(hidden_layer_sizes=(128,), max_iter=1, warm_start=True, random_state=42)
        y_tr = y_tr.astype(int)
    else:
        model = MLPRegressor(hidden_layer_sizes=(128,), max_iter=1, warm_start=True, random_state=42)

    print("Epoch | Train time (s)")
    train_times = []
    for ep in range(5):
        t0 = time.time()
        model.fit(x_tr, y_tr)
        t = time.time() - t0
        train_times.append(t)
        print(f"  {ep+1}   | {t:.4f}")

    infer_times = []
    for _ in range(5):
        t0 = time.time()
        model.predict(x_te)
        infer_times.append(time.time() - t0)

    print(f"平均训练/epoch: {np.mean(train_times):.4f}s")
    print(f"平均推理: {np.mean(infer_times):.4f}s  (样本数: {len(x_te)})")
