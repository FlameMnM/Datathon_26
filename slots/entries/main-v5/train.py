"""Fit the saved model file (per-type payout-size priors + config) from the training split."""
import argparse
import json
from pathlib import Path

import xorcast2 as xorcast

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True, help="public/ folder containing train/")
ap.add_argument("--out", default=str(Path(__file__).parent / "model.json"))
a = ap.parse_args()
q = xorcast.read_queries(Path(a.data) / "train" / "queries.csv")
h = xorcast.read_history(Path(a.data) / "train" / "history.csv")
model = {"model_version": 1, "config": xorcast.DEFAULT_CONFIG, "size_priors": xorcast.fit_size_priors(q, h)}
Path(a.out).write_text(json.dumps(model, indent=2) + "\n", encoding="utf-8")
print("wrote", a.out, model["size_priors"])
