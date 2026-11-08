"""Recreate the House Edge forecast CSV. Python 3.11+ standard library only.

python predict.py --data /data --out /output/submission.csv            # test split (default)
python predict.py --data public --out val.csv --split validation       # local scoring
"""
import argparse
from pathlib import Path

import xorcast2 as xorcast

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--split", default="test")
a = ap.parse_args()

here = Path(__file__).resolve().parent
model = xorcast.load_model(here / "model.json")
lgbm = xorcast.load_model(here / "mode_lgb.json") if (here / "mode_lgb.json").exists() else None
if (Path(a.data) / "train" / "history.csv").exists():   # learn this round's lag pairs and priors from its training split
    xorcast.learn_structure(Path(a.data) / "train", model["config"])
split = Path(a.data) / a.split
queries = xorcast.read_queries(split / "queries.csv")
history = xorcast.read_history(split / "history.csv")
xorcast.write_submission(a.out, xorcast.predict(queries, history, model, lgbm))
print(f"wrote {len(queries)} predictions to {a.out}")
