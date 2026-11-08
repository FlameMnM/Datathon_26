"""Train the size-mode classifier (LightGBM) on validation targets and export its trees to mode_lgb.json.
Needs: numpy==2.5.4, lightgbm==4.7.0 (training only; predict.py does NOT need them).
Run from this folder:  python train_mode.py --data path/to/public
"""
import argparse
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np

import xorcast2 as xc

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True)
ap.add_argument("--out", default=str(Path(__file__).parent / "mode_lgb.json"))
a = ap.parse_args()
model = xc.load_model(Path(__file__).parent / "model.json")
modes = model["config"]["size_mode"]["modes"]
split = Path(a.data) / "validation"
queries = xc.read_queries(split / "queries.csv")
history = xc.read_history(split / "history.csv")
targets = {}
import csv
with open(split / "targets.csv", newline="", encoding="utf-8") as f:
    for r in csv.DictReader(f):
        targets[r["query_id"]] = [float(r[c]) for c in ("p_1", "p_2", "p_5", "p_20")]
X, y = [], []
for q in sorted(queries, key=lambda r: r["query_id"]):
    ticks = history.get((q["episode_id"], q["machine_id"]), {})
    X.append(xc.mode_features(ticks, q["cutoff"], q["machine_type"], q["signal"], modes))
    t = np.array(targets[q["query_id"]]); t = t / t.sum()
    lo, hi = (np.array(m) / sum(m) for m in modes[q["machine_type"]])
    y.append(int(np.abs(t - hi).sum() < np.abs(t - lo).sum()))  # label: target equals the high-mode distribution
X, y = np.array(X), np.array(y)
clf = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.03, num_leaves=7, min_child_samples=20,
                         verbose=-1, n_jobs=1, random_state=0, deterministic=True)
clf.fit(X, y)
dump = clf.booster_.dump_model()


def compact(node):
    if "leaf_value" in node:
        return node["leaf_value"]
    assert node["decision_type"] == "<=", node["decision_type"]
    return [node["split_feature"], node["threshold"], compact(node["left_child"]), compact(node["right_child"])]


out = {"feature_names": xc.MODE_FEATURES, "lightgbm_version": lgb.__version__,
       "n_train": int(len(y)), "p_high_train": float(y.mean()),
       "trees": [compact(t["tree_structure"]) for t in dump["tree_info"]]}
Path(a.out).write_text(json.dumps(out) + "\n", encoding="utf-8")
# self-check: stdlib tree evaluation == lightgbm
ref = clf.predict_proba(X)[:, 1]
mine = np.array([xc.mode_lgb_prob(x, out) for x in X.tolist()])
print(f"wrote {a.out}: {len(out['trees'])} trees, labels P(high)={y.mean():.3f}, max |stdlib - lightgbm| = {np.abs(ref - mine).max():.2e}")
